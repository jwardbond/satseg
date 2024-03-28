import unittest

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils import load_data_img, create_adj, graph_to_mask


class TestUtils(unittest.TestCase):
    def test_load_data_img(self):
        """Test load_data_img function."""

        # setup
        img_path = Path(__file__).parent / "test_img.jpg"
        img_size = 280

        pil_image = Image.open(img_path).convert("RGB")
        print(pil_image.size)

        # run the function
        img_tensor, image = load_data_img(img_path, img_size)

        # test the results
        if pil_image.size[0] > pil_image.size[1]:  # if width > height
            scaled = int(pil_image.size[0] / pil_image.size[1] * img_size)
            self.assertEqual(img_tensor.shape, (1, 3, img_size, scaled))
        else:
            scaled = int(pil_image.size[1] / pil_image.size[0] * img_size)
            self.assertEqual(
                img_tensor.shape,
                (1, 3, scaled, img_size),
            )
        self.assertEqual(image.shape, (pil_image.size[1], pil_image.size[0], 3))

    def test_create_adj(self):
        features = np.random.rand(100, 384)  # NxD after feature extraction

        # Build adjacency matrix
        ccut = 1
        ncut = 0
        alpha = 3
        Wccut = create_adj(features, ccut, alpha)
        Wncut = create_adj(features, ncut, alpha)

        self.assertEqual(Wccut.shape, (100, 100))

        self.assertEqual(Wncut.shape, (100, 100))
        self.assertGreater(Wncut.max(), 0)

    def test_graph_to_mask(self):
        # Setup
        img_path = Path(__file__).parent / "test_img.jpg"
        img_size = 280
        img_tensor, image = load_data_img(img_path, img_size)

        H = img_tensor.shape[2]
        W = img_tensor.shape[3]
        P = 8
        stride = 4
        N = (1 + (H - P) // stride) * (1 + (W - P) // stride)

        S = torch.rand(N, 3)
        S = torch.softmax(S, dim=-1)
        S = torch.argmax(S, dim=-1)

        cc = False

        mask, S = graph_to_mask(S, cc, stride, P, img_tensor, image)

        self.assertEqual(mask.shape, (image.shape[0], image.shape[1]))
        self.assertEqual(S.shape, (N,))


if __name__ == "__main__":
    unittest.main(buffer=True)
