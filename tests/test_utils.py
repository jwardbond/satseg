import unittest
import itertools

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils import load_data_img, create_adj, graph_to_mask, load_data


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
        F = np.array([[1, 2], [1, 1.9], [-2, -1]])  # NxD after feature extraction

        # Build adjacency matrix
        ccut = 1
        ncut = 0
        alpha = 4

        Wccut = create_adj(F, ccut, alpha)
        Wncut = create_adj(F, ncut, alpha)

        # Correct solutions
        Wccut_correct = np.array(
            [[3.75, 3.55, -5.25], [3.55, 3.36, -5.15], [-5.25, -5.15, 3.75]]
        )

        Wncut_correct = np.array([[1, 0.96, 0], [0.96, 0.922, 0], [0, 0, 1]])

        # Test results
        self.assertEqual(Wccut.shape, (F.shape[0], F.shape[0]))
        self.assertEqual(Wncut.shape, (F.shape[0], F.shape[0]))

        self.assertTrue(np.allclose(Wccut, Wccut_correct, atol=0.01))
        self.assertTrue(np.allclose(Wncut, Wncut_correct, atol=0.01))

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

    def test_load_data(self):
        # Setup
        F = np.array([[1, 2], [1, 1.9], [-2, -1]])
        A = create_adj(F, 1, 4)

        # Create correct outputs
        edge_index_ccut = np.array(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        )
        edge_weight_ccut = np.array(
            [3.75, 3.55, -5.25, 3.55, 3.36, -5.15, -5.25, -5.15, 3.75]
        )

        # Run
        node_feats, edge_index, edge_weight = load_data(A, F)

        # Test
        self.assertEqual(node_feats.shape, F.shape)
        self.assertTrue(np.array_equal(node_feats.numpy(), F))
        self.assertTrue(np.allclose(edge_index.numpy(), edge_index_ccut))
        self.assertTrue(np.allclose(edge_weight.numpy(), edge_weight_ccut))


if __name__ == "__main__":
    unittest.main(buffer=True)
