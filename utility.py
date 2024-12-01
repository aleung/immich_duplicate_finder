from datetime import datetime
import os


def is_running_in_container():
    return os.path.exists('/.dockerenv')


def compare_and_color_data(value1, value2):
    date1 = datetime.fromisoformat(value1.rstrip('Z'))
    date2 = datetime.fromisoformat(value2.rstrip('Z'))

    # Compare the datetime objects
    if date1 > date2:  # value1 is newer
        return f"<span style='color: red;'>{value1}</span>"
    elif date1 < date2:  # value1 is older
        return f"<span style='color: green;'>{value1}</span>"
    else:  # They are the same
        return f"{value1}"


def compare_and_color(value1, value2):
    if value1 > value2:
        return f"<span style='color: green;'>{value1}</span>"
    elif value1 < value2:
        return f"<span style='color: red;'>{value1}</span>"
    else:
        return f"{value1}"
