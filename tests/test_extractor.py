import unittest
import torch
import numpy as np
from pathlib import Path
from utils import load_data_img
from satseg.extractor import ViTExtractor
from satseg.vision_transformer import VisionTransformer


class TestViTExtractor(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model_type = "facebook/dino-vits8"
        self.model_dir = (
            Path(__file__).parents[1]
            / "models"
            / "dino_deitsmall8_pretrain_full_checkpoint.pth"
        )

        self.stride = 4
        self.layer = 11
        self.facet = "key"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head = "student"

        self.D = 384  # hidden dimension size (384 is DINO ViT-S default)
        self.h = 6  # number of heads (6 is DINO ViT-S default)

    def test_create_model(self):
        """Test model creation from ViTExtractor class."""

        extractor = ViTExtractor.create_model(
            self.model_type, self.head, state_dict=self.model_dir
        )
        self.assertIsInstance(extractor, VisionTransformer)

    def test_extract_descriptors(self):
        """Test feature extraction with different strides"""

        # Model
        extractor = ViTExtractor(
            self.model_type, self.stride, self.model_dir, self.device
        )

        # Input
        img_path = Path(__file__).parent / "test_img.jpg"
        img_tensor, _ = load_data_img(img_path, 20)  # BxCxHxW

        # Test
        features = extractor.extract_descriptors(
            img_tensor.to(self.device), self.layer, self.facet
        )
        H = img_tensor.shape[2]
        W = img_tensor.shape[3]
        B = img_tensor.shape[0]
        P = extractor.model.patch_embed.patch_size
        D = self.D

        N = (1 + (H - P) // self.stride) * (1 + (W - P) // self.stride)

        deepshape = features.cpu().numpy().shape

        self.assertEqual(deepshape, (B, 1, N, D))

    def test_extract_features(self):
        """Test internal feature extraction method"""

        # Model
        extractor = ViTExtractor(
            self.model_type, self.stride, self.model_dir, self.device
        )

        # Input
        img_path = Path(__file__).parent / "test_img.jpg"
        img_tensor, _ = load_data_img(img_path, 20)  # BxCxHxW

        # Test
        output = extractor._extract_features(
            img_tensor.to(self.device), [self.layer], self.facet
        )

        B = img_tensor.shape[0]

        H = img_tensor.shape[2]
        W = img_tensor.shape[3]
        P = extractor.model.patch_embed.patch_size
        N = (1 + (H - P) // self.stride) * (1 + (W - P) // self.stride)
        N = N + 1  # at this point, we have not dropped the extra <cls> token
        D = self.D
        h = self.h

        self.assertEqual(output[0].shape, (B, h, N, D // h))


if __name__ == "__main__":
    unittest.main(buffer=True)
