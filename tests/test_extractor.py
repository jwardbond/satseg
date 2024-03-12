import unittest
from satseg.extractor import ViTExtractor


class TestViTExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ViTExtractor()

    def test_preprocess(self):
        image_path = "path/to/image.jpg"
        tensor, image = self.extractor.preprocess(image_path)
        self.assertIsNotNone(tensor)
        self.assertIsInstance(image, Image.Image)

    def test_extract_descriptors(self):
        batch = torch.randn(1, 3, 224, 224)
        descriptors = self.extractor.extract_descriptors(batch)
        self.assertIsNotNone(descriptors)
        self.assertEqual(descriptors.shape, (1, 256))

    # Add more test cases for other methods as needed
    # add more test cases
    def test_extract_features(self):
        image_path = "path/to/image.jpg"
        features = self.extractor.extract_features(image_path)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 256))

    def test_extract_features_batch(self):
        image_path = "path/to/image.jpg"
        features = self.extractor.extract_features_batch([image_path])
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 256))


if __name__ == "__main__":
    unittest.main()
