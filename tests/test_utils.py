import unittest
from PIL import Image
from pathlib import Path
from utils import load_data_img


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
        print(img_tensor.shape)
        print(image.shape)

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


if __name__ == "__main__":
    unittest.main(buffer=True)
