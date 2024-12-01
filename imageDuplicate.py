import os
import streamlit as st
import time

import torch
import numpy as np
import faiss
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from api import getImage, deleteAsset
from utility import is_running_in_container, compare_and_color, compare_and_color_data
from api import getAssetInfo
from db import load_duplicate_pairs, is_db_populated, save_duplicate_pair, delete_duplicate_pair, remove_deleted_assets_from_db
from streamlit_image_comparison import image_comparison

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load ResNet152 with pretrained weights
model = resnet152(weights=ResNet152_Weights.DEFAULT)
model.eval()  # Set model to evaluation mode


def convert_image_to_rgb(image):
    """Convert image to RGB if it's RGBA."""
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image


transform = Compose([
    convert_image_to_rgb,
    Resize((224, 224)),  # Standard size for ImageNet-trained models
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global variables for paths
data_path = 'data/' if is_running_in_container() else ''
index_path = data_path + 'faiss_index.bin'
metadata_path = data_path + 'metadata.npy'


def extract_features(image):
    """Extract features from an image using a pretrained model."""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor)
    return features.numpy().flatten()


def init_or_load_faiss_index():
    """Initialize or load the FAISS index and metadata, ensuring index is ready for use."""
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        metadata = np.load(metadata_path, allow_pickle=True).tolist()
    else:
        index = None
        metadata = []
    return index, metadata


def save_faiss_index_and_metadata(index, metadata):
    """Save the FAISS index and metadata to disk."""
    faiss.write_index(index, index_path)
    np.save(metadata_path, np.array(metadata, dtype=object))


def update_faiss_index(asset_id):
    """Update the FAISS index and metadata with a new image and its ID,
    skipping if the asset_id has already been processed."""
    global index  # Assuming index is defined globally
    index, existing_metadata = init_or_load_faiss_index()

    # Check if the asset_id is already in metadata to decide whether to skip processing
    if asset_id in existing_metadata:
        return 'skipped'  # Skip processing this image

    image = getImage(asset_id, "Thumbnail (fast)")
    if image is not None:
        features = extract_features(image)
    else:
        return 'error'

    if index is None:
        # Initialize the FAISS index with the correct dimension if it's the first time
        dimension = features.shape[0]
        index = faiss.IndexFlatL2(dimension)

    index.add(np.array([features], dtype='float32'))
    existing_metadata.append(asset_id)

    save_faiss_index_and_metadata(index, existing_metadata)
    return 'processed'


def calculateFaissIndex(assets):
    # Initialize session state variables if they are not already set
    if 'message' not in st.session_state:
        st.session_state['message'] = ""
    if 'progress' not in st.session_state:
        st.session_state['progress'] = 0
    if 'stop_index' not in st.session_state:
        st.session_state['stop_index'] = False

    # Set up the UI components
    progress_bar = st.progress(st.session_state['progress'])
    stop_button = st.button('Stop Index Processing')
    message_placeholder = st.empty()

    # Check if stop was requested and reset it if button is pressed
    if stop_button:
        st.session_state['stop_index'] = True
        st.session_state['calculate_faiss'] = False

    total_assets = len(assets)
    processed_assets = 0
    skipped_assets = 0
    error_assets = 0
    total_time = 0

    for i, asset in enumerate(assets):
        if st.session_state['stop_index']:
            st.session_state['message'] = "Processing stopped by user."
            message_placeholder.text(st.session_state['message'])
            break  # Break the loop if stop is requested

        asset_id = asset.get('id')
        start_time = time.time()

        status = update_faiss_index(asset_id)
        if status == 'processed':
            processed_assets += 1
        elif status == 'skipped':
            skipped_assets += 1
        elif status == 'error':
            error_assets += 1

        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time

        # Update progress and messages
        progress_percentage = (i + 1) / total_assets
        st.session_state['progress'] = progress_percentage
        progress_bar.progress(progress_percentage)
        estimated_time_remaining = (
            total_time / (i + 1)) * (total_assets - (i + 1))
        estimated_time_remaining_min = int(estimated_time_remaining / 60)

        st.session_state['message'] = f"Processing asset {i + 1}/{total_assets} - (Processed: {processed_assets}, Skipped: {skipped_assets}, Errors: {error_assets}). Estimated time remaining: {estimated_time_remaining_min} minutes."
        message_placeholder.text(st.session_state['message'])

    # Reset stop flag at the end of processing
    st.session_state['stop_index'] = False
    if processed_assets >= total_assets:
        st.session_state['message'] = "Processing complete!"
        message_placeholder.text(st.session_state['message'])
        progress_bar.progress(1.0)


def generate_db_duplicate():
    st.write("Database initialization")
    index, metadata = init_or_load_faiss_index()
    if not index or not metadata:
        st.write("FAISS index or metadata not available.")
        return

    # Check and update the stop mechanism in session state
    if 'stop_requested' not in st.session_state:
        st.session_state['stop_requested'] = False

    # Button to request stopping
    if st.button('Stop Finding Duplicates'):
        st.session_state['stop_requested'] = True
        st.session_state['generate_db_duplicate'] = False

    num_vectors = index.ntotal
    message_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_vectors):
        # Check if stop has been requested
        if st.session_state['stop_requested']:
            message_placeholder.text("Processing was stopped by the user.")
            progress_bar.empty()
            # Optionally, reset the stop flag here if you want the process to be restartable without refreshing the page
            st.session_state['stop_requested'] = False
            return None

        progress = (i + 1) / num_vectors
        message_placeholder.text(
            f"Finding duplicates: processing vector {i+1} of {num_vectors}")
        progress_bar.progress(progress)

        query_vector = np.array([index.reconstruct(i)])
        distances, indices = index.search(query_vector, 2)

        for j in range(1, indices.shape[1]):
            # if distances[0][j] < threshold:
            idx1, idx2 = i, indices[0][j]
            if idx1 != idx2:
                sorted_pair = (min(idx1, idx2), max(idx1, idx2))
                # Check if the indices in sorted_pair are within the bounds of metadata
                if sorted_pair[0] < len(metadata) and sorted_pair[1] < len(metadata):
                    save_duplicate_pair(
                        metadata[sorted_pair[0]], metadata[sorted_pair[1]], distances[0][j])
                else:
                    st.error(f"Metadata index out of range: {sorted_pair}")
                    # Optionally log more details or handle this case further

    message_placeholder.text(f"Finished processing {num_vectors} vectors.")
    progress_bar.empty()


######## show_duplicate_photos ########

def initialize_session_state():
    """ Initialize selected_images in session_state"""
    if 'selected_images' not in st.session_state:
        st.session_state['selected_images'] = set()
    if 'deleted_assets' not in st.session_state:
        st.session_state['deleted_assets'] = set()


def handle_deletion():
    """Handle the deletion of selected images."""
    deleted_asset_ids = []
    for asset_id in st.session_state['selected_images']:
        if deleteAsset(asset_id):
            deleted_asset_ids.append(asset_id)
        else:
            st.error(f"Failed to delete asset with ID: {asset_id}")

    if deleted_asset_ids:
        remove_deleted_assets_from_db(deleted_asset_ids)
        st.session_state['deleted_assets'].update(deleted_asset_ids)
        cleanup_after_deletion()
        st.success('Selected images have been deleted.')
        st.rerun()


def cleanup_after_deletion():
    """Clean up session state after deletion."""
    st.session_state['selected_images'].clear()
    for key in list(st.session_state.keys()):
        if key.startswith('select_'):
            del st.session_state[key]


def display_delete_form():
    """Display the deletion form with confirmation."""
    with st.form(key='delete_form'):
        st.warning(
            'Are you sure you want to delete the selected images? This action cannot be undone.')
        confirm = st.checkbox('Yes, I want to delete the selected images.')
        submit_delete = st.form_submit_button('Delete selected images')
        if submit_delete:
            if confirm:
                handle_deletion()
            else:
                st.info('Please confirm deletion by checking the box above.')


def display_selected_images():
    """Display currently selected images."""
    if st.session_state['selected_images']:
        st.write('**Selected images for deletion:**')
        for asset_id in st.session_state['selected_images']:
            st.write(f"Asset ID: {asset_id}")
        return True
    st.write('No images selected for deletion.')
    return False


def display_image_with_checkbox(column, image, asset_id):
    """
    Display an image and its associated checkbox in the specified column.

    Args:
        column: streamlit column object to display content in
        image: PIL Image object to display
        asset_id: string/int identifier for the asset
        asset_info: tuple containing asset information including original path
    """
    with column:
        # Display the image
        st.image(image, caption=f"Name: {asset_id}")

        # Create checkbox with current selection state
        selected = st.checkbox(
            f'Select for deletion (Path)',
            value=(asset_id in st.session_state['selected_images']),
            key=f'select_{asset_id}'
        )

        # Update session state based on checkbox
        if selected:
            st.session_state['selected_images'].add(asset_id)
        else:
            st.session_state['selected_images'].discard(asset_id)


def display_asset_column(col, asset1_info, asset2_info, asset_id_1, asset_id_2):
    details = f"""
    - **File name:** {asset1_info[1]}
    - **Original Path:** {asset1_info[5]}
    - **ID:** {asset_id_1}
    - **Size:** {compare_and_color(asset1_info[0], asset2_info[0])}
    - **Resolution:** {compare_and_color(asset1_info[2], asset2_info[2])}
    - **Lens Model:** {asset1_info[3]}
    - **Taken At:** {compare_and_color_data(asset1_info[4], asset2_info[4])}
    - **Is Offline:** {'Yes' if asset1_info[6] else 'No'}
    - **Is Trashed:** {'Yes' if asset1_info[7] else 'No'}
    - **Is Favorite:** {'Yes' if asset1_info[8] else 'No'}
    """
    with col:
        st.markdown(details, unsafe_allow_html=True)
        delete_button_key = f"delete-{asset_id_1}"
        delete_button_label = f"Delete"
        if st.button(delete_button_label, key=delete_button_key):
            try:
                if deleteAsset(asset_id_1):
                    st.success(f"Deleted photo {asset_id_1}")
                    st.session_state[f'deleted_photo_{asset_id_1}'] = True
                    st.session_state['show_faiss_duplicate'] = True
                    st.session_state['generate_db_duplicate'] = False
                    # remove from asset db
                    delete_duplicate_pair(asset_id_1, asset_id_2)
                else:
                    st.error(f"Failed to delete photo {asset_id_1}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(f"Failed to delete photo {asset_id_1}: {str(e)}")


def process_image_pair(asset_id_1, asset_id_2):
    """Process and display a pair of images with their information."""
    asset1_info = getAssetInfo(asset_id_1)
    asset2_info = getAssetInfo(asset_id_2)

    if not asset1_info or not asset2_info:
        st.write(
            f"Skipping pair due to missing asset info: {asset_id_1}, {asset_id_2}")
        return

    image1 = getImage(asset_id_1, 'origin')
    image2 = getImage(asset_id_2, 'origin')

    if image1 is not None and image2 is not None:
        # Proceed with image comparison
        image_comparison(
            img1=np.array(image1),
            img2=np.array(image2),
            label1=f"Name: {asset1_info[1]}",
            label2=f"Name: {asset2_info[1]}",
            width=1000,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=False,
        )

        col1, col2 = st.columns(2)
        display_image_with_checkbox(col1, image1, asset_id_1)
        display_image_with_checkbox(col2, image2, asset_id_2)
        display_asset_column(col1, asset1_info, asset2_info,
                             asset_id_1, asset_id_2)
        display_asset_column(col2, asset2_info, asset1_info,
                             asset_id_2, asset_id_1)
    else:
        st.write(
            f"Missing images for one or both assets: {asset_id_1}, {asset_id_2}")


def process_duplicate_pairs(duplicates, limit, progress_bar):
    """Process and display duplicate pairs of images with progress tracking."""
    num_duplicates_to_show = min(len(duplicates), limit)

    for i, dup_pair in enumerate(duplicates[:num_duplicates_to_show]):
        try:
            # Check if processing should be stopped
            if st.session_state.get('stop_requested', False):
                st.write("Processing was stopped by the user.")
                st.session_state['stop_requested'] = False
                st.session_state['generate_db_duplicate'] = False
                break

            # Update progress
            progress = (i + 1) / num_duplicates_to_show
            progress_bar.progress(progress)

            # Process the pair
            asset_id_1, asset_id_2 = dup_pair
            process_image_pair(asset_id_1, asset_id_2)

            st.markdown("---")  # Add separator between pairs

        except Exception as e:
            st.write(f"An error occurred while processing pair {i+1}: {e}")
            continue

    progress_bar.progress(100)


def show_duplicate_photos_faiss(limit, min_threshold, max_threshold):
    initialize_session_state()

    if not is_db_populated():
        st.write(
            "The database does not contain any duplicate entries. Please generate/update the database.")
        return

    # Load duplicates from database, excluding pairs with deleted assets
    duplicates = load_duplicate_pairs(min_threshold, max_threshold)
    duplicates = [
        dup_pair for dup_pair in duplicates
        if dup_pair[0] not in st.session_state['deleted_assets'] and dup_pair[1] not in st.session_state['deleted_assets']
    ]

    if not duplicates:
        st.write("No duplicates found.")
        return

    st.write(
        f"Found {len(duplicates)} duplicate pairs with FAISS code within threshold {min_threshold} < x < {max_threshold}:")

    if display_selected_images():
        display_delete_form()

    progress_bar = st.progress(0)
    process_duplicate_pairs(duplicates, limit, progress_bar)


# TODO: remove it
def show_duplicate_photos_faiss_old(assets, limit, min_threshold, max_threshold, immich_server_url, api_key):
    # Initialize selected_images in session_state
    if 'selected_images' not in st.session_state:
        st.session_state['selected_images'] = set()
    if 'deleted_assets' not in st.session_state:
        st.session_state['deleted_assets'] = set()

    # First check if the database is populated
    if not is_db_populated():
        st.write(
            "The database does not contain any duplicate entries. Please generate/update the database.")
        return  # Exit the function early if the database is not populated

    # Load duplicates from database, excluding pairs with deleted assets
    duplicates = load_duplicate_pairs(min_threshold, max_threshold)
    duplicates = [
        dup_pair for dup_pair in duplicates
        if dup_pair[0] not in st.session_state['deleted_assets'] and dup_pair[1] not in st.session_state['deleted_assets']
    ]

    if duplicates:
        st.write(
            f"Found {len(duplicates)} duplicate pairs with FAISS code within threshold {min_threshold} < x < {max_threshold}:")

        # Add buttons and delete functionality at the top
        col_select_all, col_deselect_all, col_delete = st.columns(3)
        with col_select_all:
            if st.button('Select all images in /libraries/Recovered'):
                # Add all images in /libraries/Recovered to selected_images
                images_selected = False  # Flag to track if any images were selected
                for dup_pair in duplicates[:limit]:
                    asset_id_1, asset_id_2 = dup_pair
                    asset1_info = getAssetInfo(asset_id_1, assets)
                    asset2_info = getAssetInfo(asset_id_2, assets)

                    # Handle asset1_info
                    if asset1_info:
                        original_path_1 = asset1_info[5]
                        if original_path_1.startswith('/libraries/Recovered'):
                            st.session_state['selected_images'].add(asset_id_1)
                            images_selected = True
                    else:
                        st.write(
                            f"Asset info not found for asset_id: {asset_id_1}")

                    # Handle asset2_info
                    if asset2_info:
                        original_path_2 = asset2_info[5]
                        if original_path_2.startswith('/libraries/Recovered'):
                            st.session_state['selected_images'].add(asset_id_2)
                            images_selected = True
                    else:
                        st.write(
                            f"Asset info not found for asset_id: {asset_id_2}")

                if not images_selected:
                    st.info("No duplicates found in /libraries/Recovered.")

        with col_deselect_all:
            if st.button('Deselect all images in /libraries/Recovered'):
                # Remove all images in /libraries/Recovered from selected_images
                images_deselected = False  # Flag to track if any images were deselected
                for dup_pair in duplicates[:limit]:
                    asset_id_1, asset_id_2 = dup_pair
                    asset1_info = getAssetInfo(asset_id_1, assets)
                    asset2_info = getAssetInfo(asset_id_2, assets)

                    # Handle asset1_info
                    if asset1_info:
                        original_path_1 = asset1_info[5]
                        if original_path_1.startswith('/libraries/Recovered'):
                            st.session_state['selected_images'].discard(
                                asset_id_1)
                            images_deselected = True
                    else:
                        st.write(
                            f"Asset info not found for asset_id: {asset_id_1}")

                    # Handle asset2_info
                    if asset2_info:
                        original_path_2 = asset2_info[5]
                        if original_path_2.startswith('/libraries/Recovered'):
                            st.session_state['selected_images'].discard(
                                asset_id_2)
                            images_deselected = True
                    else:
                        st.write(
                            f"Asset info not found for asset_id: {asset_id_2}")

                if not images_deselected:
                    st.info("No images to deselect in /libraries/Recovered.")

        # Display selected images at the top
        if st.session_state['selected_images']:
            st.write('**Selected images for deletion:**')
            for asset_id in st.session_state['selected_images']:
                st.write(f"Asset ID: {asset_id}")

            # Use a form to group delete button and confirmation checkbox
            with st.form(key='delete_form'):
                st.warning(
                    'Are you sure you want to delete the selected images? This action cannot be undone.')
                confirm = st.checkbox(
                    'Yes, I want to delete the selected images.')
                submit_delete = st.form_submit_button('Delete selected images')
                if submit_delete:
                    if confirm:
                        # Proceed with deletion
                        deleted_asset_ids = []
                        for asset_id in st.session_state['selected_images']:
                            deletion_result = deleteAsset(
                                immich_server_url, asset_id, api_key)
                            if deletion_result:
                                deleted_asset_ids.append(asset_id)
                            else:
                                st.error(
                                    f"Failed to delete asset with ID: {asset_id}")
                        if deleted_asset_ids:
                            # Update the duplicates database
                            remove_deleted_assets_from_db(deleted_asset_ids)
                            # Update deleted assets in session state
                            st.session_state['deleted_assets'].update(
                                deleted_asset_ids)
                        st.success('Selected images have been deleted.')
                        st.session_state['selected_images'].clear()
                        # Reset individual checkbox states
                        for key in list(st.session_state.keys()):
                            if key.startswith('select_'):
                                del st.session_state[key]
                        # Refresh the display by rerunning the function
                        st.rerun()
                    else:
                        st.info(
                            'Please confirm deletion by checking the box above.')
        else:
            st.write('No images selected for deletion.')

        progress_bar = st.progress(0)
        num_duplicates_to_show = min(len(duplicates), limit)

        for i, dup_pair in enumerate(duplicates[:num_duplicates_to_show]):
            try:
                # Check if stop was requested
                if st.session_state.get('stop_requested', False):
                    st.write("Processing was stopped by the user.")
                    # Reset the flag for future operations
                    st.session_state['stop_requested'] = False
                    st.session_state['generate_db_duplicate'] = False
                    break  # Exit the loop

                progress = (i + 1) / num_duplicates_to_show
                progress_bar.progress(progress)

                asset_id_1, asset_id_2 = dup_pair

                asset1_info = getAssetInfo(asset_id_1, assets)
                asset2_info = getAssetInfo(asset_id_2, assets)

                # Skip if asset info is missing
                if not asset1_info or not asset2_info:
                    st.write(
                        f"Skipping pair due to missing asset info: {asset_id_1}, {asset_id_2}")
                    continue

                image1 = getImage(
                    asset_id_1, immich_server_url, 'origin', api_key)
                image2 = getImage(
                    asset_id_2, immich_server_url, 'origin', api_key)

                if image1 is not None and image2 is not None:
                    # Convert PIL images to numpy arrays if necessary
                    image1_array = np.array(image1)
                    image2_array = np.array(image2)

                    # Proceed with image comparison
                    image_comparison(
                        img1=image1_array,
                        img2=image2_array,
                        label1=f"Name: {asset_id_1}",
                        label2=f"Name: {asset_id_2}",
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=False,
                    )

                    col1, col2 = st.columns(2)

                    # Display first image and its checkbox
                    with col1:
                        st.image(image1, caption=f"Name: {asset_id_1}")
                        # original_path is the 6th item
                        original_path_1 = asset1_info[5]
                        checkbox_key_1 = f'select_{asset_id_1}'
                        # Set checkbox value based on whether asset_id_1 is in selected_images
                        selected_1 = st.checkbox(
                            f'Select for deletion',
                            value=(
                                asset_id_1 in st.session_state['selected_images']),
                            key=checkbox_key_1
                        )
                        if selected_1:
                            st.session_state['selected_images'].add(asset_id_1)
                        else:
                            st.session_state['selected_images'].discard(
                                asset_id_1)

                    # Display second image and its checkbox
                    with col2:
                        st.image(image2, caption=f"Name: {asset_id_2}")
                        original_path_2 = asset2_info[5]
                        checkbox_key_2 = f'select_{asset_id_2}'
                        selected_2 = st.checkbox(
                            f'Select for deletion (Path: {original_path_2})',
                            value=(
                                asset_id_2 in st.session_state['selected_images']),
                            key=checkbox_key_2
                        )
                        if selected_2:
                            st.session_state['selected_images'].add(asset_id_2)
                        else:
                            st.session_state['selected_images'].discard(
                                asset_id_2)

                    display_asset_column(
                        col1, asset1_info, asset2_info, asset_id_1, asset_id_2, immich_server_url, api_key)
                    display_asset_column(
                        col2, asset2_info, asset1_info, asset_id_2, asset_id_1, immich_server_url, api_key)

                else:
                    st.write(
                        f"Missing images for one or both assets: {asset_id_1}, {asset_id_2}")

                st.markdown("---")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        progress_bar.progress(100)
    else:
        st.write("No duplicates found.")
