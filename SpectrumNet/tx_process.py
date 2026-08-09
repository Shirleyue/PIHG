import json
import pdb
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
from PIL import Image
import os


def load_tx_positions(file_path):
    """
    Read the transmitter positions from a .txt file, returned as [(x1,y1), (x2,y2), ...].
    :param file_path: file path; each line is either "(x,y)" or "x,y"
    :return: list of transmitter coordinates
    """
    tx_positions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            # Both formats are accepted: (x,y) and x,y
            if line.startswith('(') and line.endswith(')'):
                x, y = map(int, line[1:-1].split(','))
            else:
                x, y = map(int, line.split(','))
            tx_positions.append((x, y))
    return tx_positions


if __name__ == "__main__":

    txt_info = "/root/autodl-tmp/SpectrumNet/tx_info.txt"  # file holding the npz mapping dict
    df = pd.read_csv(txt_info, sep='\t', header=None, names=['filename', 'info'])
    folder_pth = '/root/autodl-tmp/SpectrumNet/tx/'
    folder_dB = '/root/autodl-tmp/SpectrumNet/tx_dB/'

    os.makedirs(folder_pth, exist_ok=True)
    os.makedirs(folder_dB, exist_ok=True)


    print(df.head())
    df['info'] = df['info'].apply(ast.literal_eval)
    power_max = 100
    for _, row in tqdm(df.iterrows(), total=1):
        file_name = row['filename']
        info = row['info']
        # Write the transmitter coordinates to a txt file, one "(x,y)" per line
        tx_coords = []
        tx_img = np.zeros((128, 128), dtype=float)

        with open(os.path.join(folder_pth, f"{file_name}_coords.txt"), 'w') as f:
            for tx in info:
                x, y, power = round(tx['x']), round(tx['y']), tx['power']
                if power > power_max:
                    power_max = power
                if 0 <= x < 128 and 0 <= y < 128:
                    f.write(f"({x},{y})\n")
                    tx_coords.append((x, y))    # x is the column index, y the row index
                    tx_img[y, x] = power

                else:
                    print(f"Warning: {file_name} has invalid coordinate ({x},{y})")
        np.save(os.path.join(folder_dB, f"{file_name}.npy"), tx_img)

    print('power_max:', power_max)  # 999.8

    tx_positions = load_tx_positions("/root/autodl-tmp/SpectrumNet/tx/T01C0D0000_n00_coords.txt")
    print(tx_positions)  # e.g. [(5, 7), (8, 2), (6, 12)]
    tx_power = np.load("/root/autodl-tmp/SpectrumNet/tx_dB/T01C0D0000_n00.npy")
    print(tx_power)
    print('Done!')
