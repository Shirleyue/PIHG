'''
code by jiangxinyue(shirleyuue@foxmail.com)
date:2025-03-10
'''


import numpy as np
import matplotlib.pyplot as plt
import re

def parse_npz_filename(filename):
    """
    Parse the parameters encoded in a file name, e.g. T06C2D0004_n02_bdtr.npz
    - Txx: terrain type (11 of them, e.g. T06 = forest)
    - Cx: climate type (3 of them, C0 = tropical, C1 = subtropical, C2 = temperate)
    - Dxxxx: map id (0000-9999)
    - nxx: sample id (00-99)
    - bdtr: fixed marker (probably standing for building and terrain data)
    """
    pattern = r"T(\d{2})C(\d)D(\d{4})_n(\d{2})_bdtr\.npz"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"invalid file name format: {filename}")

    params = {
        "terrain_type": int(match.group(1)),  # terrain-type code
        "climate_type": int(match.group(2)),  # climate-type code
        "map_id": match.group(3),             # map id
        "sample_id": int(match.group(4))      # sample id
    }
    return params

def load_npz_data(filepath):
    """Load the .npz file and parse its metadata."""
    filename = filepath.split("/")[-1]

    # 1. Parse the file-name parameters
    try:
        params = parse_npz_filename(filename)
    except ValueError as e:
        print(f"[error] failed to parse the file name: {e}")
        return None

    # 2. Load the .npz file
    try:
        data = np.load(filepath)
        array_names = data.files
        arrays = {name: data[name] for name in array_names}
        data.close()
    except Exception as e:
        print(f"[error] failed to load the .npz file: {e}")
        return None

    # 3. Attach the metadata
    metadata = {
        "terrain_type": _get_terrain_name(params["terrain_type"]),
        "climate": _get_climate_name(params["climate_type"]),
        "map_id": params["map_id"],
        "sample_id": params["sample_id"],
        "arrays": arrays  # dictionary holding every array
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

# --------------- example usage ---------------
if __name__ == "__main__":
    filepath = "./data/npz/T04C0D0017_n00_bdtr.npz"
    data = load_npz_data(filepath)

    if data is not None:
        print("Parsed metadata:")
        print(f"- terrain type: {data['terrain_type']}")
        print(f"- climate: {data['climate']}")
        print(f"- map id: {data['map_id']}")
        print(f"- sample id: {data['sample_id']}")

        print("\nArray info:")
        for name, array in data["arrays"].items():
            print(f"- {name}: shape={array.shape}, dtype={array.dtype}")

        # Visualise a couple of the arrays, when terrain and building data are present
        if "terrain_yx" in data["arrays"]:
            plt.figure(figsize=(6, 6))
            plt.imshow(data["arrays"]["terrain_yx"], cmap="terrain")
            plt.title(f"Terrain Map - {data['terrain_type']}")
            plt.colorbar(label="Elevation")
            plt.show()

        if "inBldg_zyx" in data["arrays"]:
            for z in range(data["arrays"]["inBldg_zyx"].shape[0]):
                plt.figure(figsize=(6, 6))
                plt.imshow(data["arrays"]["inBldg_zyx"][z], cmap="binary")
                plt.title(f"Building Map (Layer {z}) - {data['terrain_type']}")
                plt.colorbar(label="Building Presence (0/1)")
                plt.show()
