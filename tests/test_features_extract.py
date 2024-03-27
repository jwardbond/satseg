import unittest
from pathlib import Path

import torch
from satseg.extractor import ViTExtractor
from satseg.features_extract import deep_features
from utils import load_data_img


class TestFeaturesExtract(unittest.TestCase):
    def test_deep_feature_shape(self):
        # Create model
        model_type = "facebook/dino-vits8"
        model_dir = (
            Path(__file__).parents[1]
            / "models"
            / "dino_deitsmall8_pretrain_full_checkpoint.pth"
        )
        stride = 8
        device = "cuda" if torch.cuda.is_available() else "cpu"

        extractor = ViTExtractor(model_type, stride, model_dir, device)

        # Create input
        img_path = Path(__file__).parent / "test_img.jpg"
        img_tensor, _ = load_data_img(img_path, 280)  # 1xCxHxW

        layer = 11
        facet = "key"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Construct tensors
        B = img_tensor.shape[0]
        H = img_tensor.shape[2]
        W = img_tensor.shape[3]
        P = extractor.model.patch_embed.patch_size
        N = (1 + (H - P) // stride) * (1 + (W - P) // stride)

        D = 384  # default hidden dimension size for ViT-S DiNO

        # Get output
        features = deep_features(img_tensor, extractor, layer, facet, device=device)

        # Test
        self.assertEqual(features.shape, (B * N, D))
