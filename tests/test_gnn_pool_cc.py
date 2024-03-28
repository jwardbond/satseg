import unittest

import numpy as np
from pathlib import Path
from torch_geometric.data import Data

import torch
from satseg.gnn_pool_cc import GNNpool
from utils import load_data_img, create_adj, load_data


class TestGNN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = 3
        self.conv_hidden = 3
        self.mlp_hidden = 3
        self.num_clusters = 2
        self.device = "cpu"

    def test_create_model(self):
        """Test model creation from GNNPool class."""

        gnn = GNNpool(
            self.input_dim,
            self.conv_hidden,
            self.mlp_hidden,
            self.num_clusters,
            self.device,
        )
        self.assertIsInstance(gnn, GNNpool)

    def test_forward(self):
        """Test forward pass of GNNPool class."""

        # Create dummy input
        image_tensor, _ = load_data_img(
            Path(__file__).parent / "test_img.jpg", image_size=50
        )
        image = image_tensor[0].permute(1, 2, 0).numpy()
        F = image.reshape(-1, 3)
        N = F.shape[0]

        W = create_adj(F, 1, 3)

        node_feats, edge_index, edge_weight = load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(self.device)

        # Create model
        gnn = GNNpool(
            self.input_dim,
            self.conv_hidden,
            self.mlp_hidden,
            self.num_clusters,
            self.device,
        )

        # Forward pass
        a, s = gnn(data, torch.from_numpy(W).to(self.device))

        # Test
        self.assertEqual(s.shape, (N, self.num_clusters))
        self.assertEqual(a.shape, torch.from_numpy(W).shape)


if __name__ == "__main__":
    unittest.main()
