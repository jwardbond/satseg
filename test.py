from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch


in_dir = Path("./input_images/deepglobe/custom/images/")

for item in tqdm(
    list(in_dir.glob("*.jpg"))
    + list(in_dir.glob("*.png"))
    + list(in_dir.glob("*.jpeg"))
):
    print(item)
