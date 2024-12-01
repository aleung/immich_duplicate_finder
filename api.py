import requests
import json
import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageFile
from io import BytesIO
from db import bytes_to_megabytes
from pillow_heif import register_heif_opener
import os

# module scope
assets = []
base_url = None
api_key = None
timeout = None


def api_init(immich_server_url, immich_api_key, api_timeout):
    global base_url, api_key, timeout

    # Remove trailing slash from immich_server_url if present
    base_url = immich_server_url.rstrip('/') + '/api'
    api_key = immich_api_key
    timeout = api_timeout


@st.cache_data(show_spinner=True)
def fetchAssets(type):
    global assets

    # Initialize messaging and progress
    if 'fetch_message' not in st.session_state:
        st.session_state['fetch_message'] = ""
    message_placeholder = st.empty()

    # Initialize assets to None or an empty list, depending on your usage expectation
    assets = []

    asset_paths_url = f"{base_url}/view/folder/unique-paths"
    asset_info_url = f"{base_url}/view/folder"

    try:
        with st.spinner('Fetching assets...'):
            # Fetch unique paths
            response = requests.get(asset_paths_url, headers={
                                    'Accept': 'application/json', 'x-api-key': api_key}, verify=False, timeout=timeout)
            response.raise_for_status()  # This will raise an exception for HTTP errors

            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                if response.text:
                    paths = response.json()  # Decode JSON response into a list of paths
                else:
                    st.session_state['fetch_message'] = 'Received an empty response.'
                    paths = []  # Set paths to empty list if response is empty
            else:
                st.session_state[
                    'fetch_message'] = f'Unexpected Content-Type: {content_type}\nResponse content: {response.text}'
                paths = []  # Set paths to empty list if unexpected content type

            # Fetch assets for each path
            for path in paths:
                if path:
                    response = requests.get(f'{asset_info_url}?path={path}', headers={
                                            'Accept': 'application/json', 'x-api-key': api_key}, verify=False, timeout=timeout)
                    response.raise_for_status()

                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        if response.text:
                            # Add assets from the current path
                            assets.extend(response.json())
                        else:
                            st.session_state['fetch_message'] = 'Received an empty response for path.'
                    else:
                        st.session_state[
                            'fetch_message'] = f'Unexpected Content-Type for path: {content_type}\nResponse content: {response.text}'

            # Filter assets by type
            assets = [asset for asset in assets if asset.get("type") == type]
            st.session_state['fetch_message'] = 'Assets fetched successfully!'

    except requests.exceptions.ConnectTimeout:
        st.session_state['fetch_message'] = 'Failed to connect to the server. Please check your network connection and try again.'
        assets = []  # Set assets to empty list on connection timeout

    except requests.exceptions.HTTPError as e:
        st.session_state['fetch_message'] = f'HTTP error occurred: {e}'
        assets = []  # Set assets to empty list on HTTP error

    except requests.exceptions.RequestException as e:
        st.session_state['fetch_message'] = f'Error fetching assets: {e}'
        assets = []  # Set assets to empty list on other request errors

    message_placeholder.text(st.session_state['fetch_message'])
    return assets


def getImage(asset_id, photo_choice):
    # Determine whether to fetch the original or thumbnail based on user selection
    register_heif_opener()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if photo_choice == 'Thumbnail (fast)':
        response = requests.request("GET", f"{base_url}/assets/{asset_id}/thumbnail?size=thumbnail", headers={
                                    'Accept': 'application/octet-stream', 'x-api-key': api_key}, data={})
    else:
        response = requests.request("GET", f"{base_url}/assets/{asset_id}/original", headers={
                                    'Accept': 'application/octet-stream', 'x-api-key': api_key}, data={})

    if response.status_code == 200 and 'image/' in response.headers.get('Content-Type', ''):
        image_bytes = BytesIO(response.content)
        try:
            image = Image.open(image_bytes)
            image.load()  # Force loading the image data while the file is open
            image_bytes.close()  # Now we can safely close the stream
            return image
        except UnidentifiedImageError:
            print(
                f"Failed to identify image for asset_id {asset_id}. Content-Type: {response.headers.get('Content-Type')}")
            image_bytes.close()  # Ensure the stream is closed even if an error occurs
            return None
        finally:
            image_bytes.close()  # Ensure the stream is always closed
            del image_bytes
    else:
        print(
            f"Skipping non-image asset_id {asset_id} with Content-Type: {response.headers.get('Content-Type')}")
        return None


@st.cache_data
def fetch_asset_info(asset_id):
    asset_info_url = f"{base_url}/assets/{asset_id}"
    response = requests.get(asset_info_url, headers={
                            'Accept': 'application/json', 'x-api-key': api_key}, verify=False, timeout=timeout)
    response.raise_for_status()  # This will raise an exception for HTTP errors
    return response.json()


def getAssetInfo(asset_id):
    # Search for the asset in the provided list of assets.
    asset_info = next(
        (asset for asset in assets if asset['id'] == asset_id), None)

    if not asset_info:
        asset_info = fetch_asset_info(asset_id)

    if asset_info:
        # Extract all required info.
        try:
            formatted_file_size = bytes_to_megabytes(
                asset_info['exifInfo']['fileSizeInByte'])
        except KeyError:
            formatted_file_size = "Unknown"

        original_file_name = asset_info.get('originalFileName', 'Unknown')
        resolution = "{} x {}".format(
            asset_info.get('exifInfo', {}).get('exifImageHeight', 'Unknown'),
            asset_info.get('exifInfo', {}).get('exifImageWidth', 'Unknown')
        )
        lens_model = asset_info.get('exifInfo', {}).get('lensModel', 'Unknown')
        creation_date = asset_info.get('fileCreatedAt', 'Unknown')
        date_time_original = asset_info.get(
            'exifInfo', {'dateTimeOriginal': creation_date}).get('dateTimeOriginal', 'Unknown')
        original_path = asset_info.get('originalPath', 'Unknown')
        is_offline = asset_info.get('isOffline', False)
        is_trashed = asset_info.get('isTrashed', False)  # Extract isTrashed
        is_favorite = asset_info.get('isFavorite', False)
        # Add more fields as needed and return them
        return formatted_file_size, original_file_name, resolution, lens_model, date_time_original, original_path, is_offline, is_trashed, is_favorite
    else:
        return None


def deleteAsset(asset_id):
    st.session_state['show_faiss_duplicate'] = False
    url = f"{base_url}/assets"
    payload = json.dumps({
        "force": True,
        "ids": [asset_id]
    })
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    try:
        response = requests.delete(url, headers=headers, data=payload)
        if response.status_code == 204:
            st.write(f"Successfully deleted asset with ID: {asset_id}")
            print(f"Successfully deleted asset with ID: {asset_id}")
            return True
        else:
            # Provide more detailed error feedback
            try:
                error_message = response.json().get(
                    'message', 'No additional error message provided.')
            except ValueError:
                error_message = 'Response content is not valid JSON.'
            st.error(
                f"Failed to delete asset with ID: {asset_id}. Status code: {response.status_code}. Message: {error_message}")
            print(
                f"Failed to delete asset with ID: {asset_id}. Status code: {response.status_code}. Message: {error_message}")
            return False
    except requests.RequestException as e:
        # Handle request-related exceptions
        st.error(f"Request failed: {str(e)}")
        print(f"Request failed: {str(e)}")
        return False


def updateAsset(asset_id, dateTimeOriginal, description, isFavorite, latitude, longitude, isArchived):
    # Ensure the URL is constructed correctly
    url = f"{base_url}/assets/{asset_id}"

    payload = json.dumps({
        "dateTimeOriginal": dateTimeOriginal,
        "description": description,
        "isArchived": isArchived,
        "isFavorite": isFavorite,
        "latitude": latitude,
        "longitude": longitude
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': api_key  # Authorization via API key
    }

    try:
        response = requests.put(url, headers=headers, data=payload)
        if response.status_code == 200:
            response_data = response.json()
            st.success(
                f"Successfully move on archive asset with ID: {asset_id}")
            print(
                f"Successfully move on archive asset with ID: {asset_id}. Response: {response_data}")
            return True
        else:
            error_message = response.json().get(
                'message', 'No additional error message provided.')
            st.error(
                f"Failed to move on archive asset with ID: {asset_id}. Status code: {response.status_code}. Message: {error_message}")
            print(
                f"Failed to move on archive asset with ID: {asset_id}. Status code: {response.status_code}. Message: {error_message}")
            return False
    except requests.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        print(f"Request failed: {str(e)}")
        return False

# For video function


def getVideoAndSave(asset_id, save_directory):
    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    response = requests.get(f"{base_url}/download/assets/{asset_id}", headers={
                            'Accept': 'application/octet-stream', 'x-api-key': api_key}, stream=True)
    file_path = os.path.join(save_directory, f"{asset_id}.mp4")

    if response.status_code == 200 and 'video/' in response.headers.get('Content-Type', ''):
        try:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        except Exception as e:
            print(f"Failed to save video for asset_id {asset_id}. Error: {e}")
            return None
    else:
        print(
            f"Failed to retrieve video for asset_id {asset_id}. Status Code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
        return None
