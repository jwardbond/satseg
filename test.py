from pathlib import Path
import numpy as np
import torch

adj = np.random.rand(3, 2, 2)
print(adj.reshape(-1, 3).shape)
