'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re

def parse_png_filename(filename):
    """
    Parse the parameters encoded in a file name, e.g. T01C0D0000_n00_f01_ss_z00.png
    - Txx: terrain type (11 of them, e.g. T01 = dense urban)
    - Cx: climate type (3 of them, C0 = tropical, C1 = subtropical, C2 = temperate)
    - Dxxxx: map id (0000-9999)
    - nxx: sample id (00-99)
    - fxx: frequency id (00-04, one per band, e.g. f00 = 150MHz)
    - zxx: height layer (00 = 1.5m, 01 = 30m, 02 = 200m)
    - ss: fixed marker (probably standing for signal strength map)
    """
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_f(\d{2})_ss_z(\d{2})\.png"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"invalid file name format: {filename}")

    params = {
        "terrain_type": int(match.group(1)),  # terrain-type code
        "climate_type": int(match.group(2)),  # climate-type code
        "map_id": match.group(3),             # map id
        "sample_id": int(match.group(4)),     # sample id
        "frequency_id": int(match.group(5)),  # frequency id
        "height_id": int(match.group(6))      # height-layer id
    }
    return params

def load_png_data(filepath):
    """Load the image file and parse its metadata."""
    filename = filepath.split("/")[-1]

    # 1. Parse the file-name parameters
    try:
        params = parse_png_filename(filename)
    except ValueError as e:
        print(f"[error] failed to parse the file name: {e}")
        return None

    # 2. Load the image data
    try:
        img = Image.open(filepath)
        img_array = np.array(img)  # as a numpy array (H, W, C)
    except Exception as e:
        print(f"[error] failed to load the image: {e}")
        return None

    # 3. Attach the metadata
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "frequency": _get_frequency(params["frequency_id"]),
        "height": _get_height(params["height_id"]),
        "data": img_array  # image data (128x128x3)
    }
    return metadata

# --------------- helpers: turn codes into readable text ---------------
def _get_terrain_name(code):
    """Terrain-type code to name."""
    terrain_map = {
        1: "Dense Urban",
        2: "Ordinary Urban",
        3: "Rural",
        4: "Suburban",
        5: "Mountainous",
        6: "Forest",
        7: "Desert",
        8: "Grassland",
        9: "Island",
        10: "Ocean",
        11: "Lake"
    }
    return terrain_map.get(code, "Unknown")

def _get_climate_name(code):
    """Climate-type code to name."""
    climate_map = {
        0: "Tropical",
        1: "Subtropical",
        2: "Temperate"
    }
    return climate_map.get(code, "Unknown")

def _get_frequency(code):
    """Frequency id to the actual value (MHz)."""
    freq_map = {
        0: 150,    # 150 MHz
        1: 1500,   # 1.5 GHz
        2: 1700,   # 1.7 GHz
        3: 3500,   # 3.5 GHz
        4: 22000   # 22 GHz
    }
    return freq_map.get(code, -1)


def _get_height(code):
    """Height-layer id to the actual value (metres)."""
    height_map = {
        0: 1.5,
        1: 30.0,
        2: 200.0
    }
    return height_map.get(code, -1.0)

# --------------- example usage ---------------
if __name__ == "__main__":
    filepath = "T01C0D0000_n00_f01_ss_z00.png"

    data = load_png_data(filepath)

    if data is not None:
        print("Parsed metadata:")
        print(f"- terrain type: {data['terrain_type']}")
        print(f"- climate: {data['climate']}")
        print(f"- frequency: {data['frequency']} MHz")
        print(f"- height: {data['height']} m")
        print(f"- image shape: {data['data'].shape}")

        plt.figure(figsize=(6, 6))
        plt.imshow(data['data'])
        plt.title(f"{data['terrain_type']} - {data['height']}m - {data['frequency']}MHz")
        plt.axis('off')
        plt.show()
