import argparse
import pathlib
from collections import defaultdict

import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
from satseg.extractor import ViTExtractor
from satseg.features_extract import deep_features

pathlib.PosixPath = pathlib.WindowsPath
##########################################################################################
# Adapted from https://github.com/SAMPL-Weizmann/DeepCut

# Note that for all code herein, the following convention is used:
#   D is size of deep feature vector
#   B is batch size
#   C is the number of channels
#   H is image height
#   W is image width
#   P is patch size
#   N is the number of tokens
#       [1+(H-P)//stride] * [1+(W-P)//stride]
#       + 1 (if include_cls)
#   h is the number of attention heads
##########################################################################################


def get_saliency_map(
    device: str,
    out_dir: pathlib.PurePath,
    save: bool,
    pretrained_weights: pathlib.PurePath,
    res: tuple[int, int],
    stride: int,
    imgpath: pathlib.PurePath,
    **kwargs,
):
    """Generates a saliency map for an image using attention between [CLS] and all other patches

    Args:
        device: Device to use ('cuda'/'cpu')
        out_dir: Filepath for results
        save: True to save results, else false
        pretrained_weights: Path to pretrained ViT
        res: Resolution of image
        stride: Stride for feature extraction
        imgpath: Path to input image
    """

    # Set up tmp directory
    tmp_dir = pathlib.Path("./.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ##########################################################################################
    # Init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )

    image_tensor, image = utils.load_data_img(imgpath, res)

    ################################################################################################
    # Extract saliency
    ################################################################################################
    # for head in range(0, 6):
    saliency = extractor.extract_saliency_maps(
        image_tensor.to(device),  # head_idxs=[head]
    )

    # Reshape
    P = extractor.model.patch_embed.patch_size

    saliency = saliency.detach().cpu()
    saliency = saliency.squeeze(0)
    saliency = np.array(
        torch.reshape(
            saliency,
            (
                int(1 + (image_tensor.shape[2] - P) // stride),  # height in # patches
                int(1 + (image_tensor.shape[3] - P) // stride),  # width in # patches
            ),
        )
    )  # HxW (in patches)

    img = cv2.resize(
        saliency.astype("float"),
        (image[:, :, 0].shape[1], image[:, :, 0].shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Save or show
    if save:
        plt.imsave(
            out_dir / (imgpath.stem + f"{pretrained_weights.stem}_saliency" + ".png"),
            dpi=200,
            arr=img,
        )
    else:
        plt.imshow(img)
        plt.show()


def patch_similarity(
    device: str,
    out_dir: pathlib.PurePath,
    save: bool,
    pretrained_weights: pathlib.PurePath,
    res: tuple[int, int],
    layer: int,
    facet: str,
    stride: int,
    source_image: pathlib.PurePath,
    target_image: pathlib.PurePath,
    source_patch: int,
    **kwargs,
):
    """Computes cosine similarity between two images using source patches key/query/value/output token

    Args:
        device: Device to use ('cuda'/'cpu')
        out_dir: Filepath for results
        save: True to save results, else false
        pretrained_weights: Path to pretrained ViT
        res: Resolution of image
        stride: Stride for feature extraction
    """

    ##########################################################################################
    # Init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )

    source_tensor, source = utils.load_data_img(source_image, res)
    target_tensor, target = utils.load_data_img(target_image, res)

    ##########################################################################################
    # Process
    ##########################################################################################
    source_F = deep_features(source_tensor, extractor, layer, facet, device=device)
    target_F = deep_features(target_tensor, extractor, layer, facet, device=device)

    output = np.dot(target_F, source_F[source_patch].T)  # NxD x Dx1

    # Reshape to HxW (in patches)
    P = extractor.model.patch_embed.patch_size
    output = np.reshape(
        output,
        (
            int(1 + (target_tensor.shape[2] - P) // stride),  # height in # patches
            int(1 + (target_tensor.shape[3] - P) // stride),  # width in # patches
        ),
    )

    img = cv2.resize(
        output.astype("float"),
        (target[:, :, 0].shape[1], target[:, :, 0].shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Save or show
    if save:
        plt.imsave(
            out_dir / (imgpath.stem + f"{pretrained_weights.stem}_saliency" + ".png"),
            dpi=200,
            arr=[source, img],
        )
    else:
        utils.im_show_n([source, img], 2, "Source, Target")
        plt.imshow(img)
        plt.show()


def process_config(confpath):
    """Loads and processes the config file for running satseg"""

    # Loading
    with open(confpath) as f:
        raw = yaml.safe_load(f)
    config = defaultdict(None, raw)

    # Processing
    config["device"] = config["device"] if torch.cuda.is_available() else "cpu"
    config["res"] = tuple(config["res"])

    assert not (
        (config["K"] != 2) and (config["cc"])
    ), "largest connected component only available for k == 2"

    if config["cut"] == 1:  # If Correlational Clustering, set max # clusters
        config["K"] = 10

    config["pretrained_weights"] = pathlib.Path(config["pretrained_weights"])

    # Check if model, download if not
    model_folder = pathlib.Path(config["pretrained_weights"].parents[0])
    model_folder.mkdir(parents=True, exist_ok=True)

    if not config["pretrained_weights"].exists():
        if "dino_deitsmall8" in config["pretrained_weights"].stem:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth"
            utils.download_url(url, config["pretrained_weights"])
        elif "vit_mc" in config["pretrained_weights"].stem:
            print(
                "Please download the model from https://github.com/WennyXY/DINO-MC?tab=readme-ov-file"
            )
            exit(1)
        else:
            print(f"Model not found at {config['pretrained_weights']}")
            exit(1)
    # Set input and output directories
    config["in_dir"] = pathlib.Path(config["in_dir"])
    config["out_dir"] = pathlib.Path(config["out_dir"])

    return config, raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="saliency",
        description="Generates a saliency map according to https://dino-vit-features.github.io/paper.pdf",
    )
    parser.add_argument("confpath")
    parser.add_argument("filepath", help="Path to input image")

    args = parser.parse_args()
    config, raw_config = process_config(args.confpath)

    if config["save"]:
        # Create specific output dir
        filepath = pathlib.Path(args.filepath)
        config["out_dir"] = config["out_dir"] / filepath.stem
        config["out_dir"].mkdir(parents=True, exist_ok=True)

        # Save raw config file
        with open(config["out_dir"] / "config.yml", "w") as f:
            yaml.dump(raw_config, f, sort_keys=False)

    # Run
    imgpath = pathlib.Path(args.filepath)
    get_saliency_map(**config, imgpath=imgpath)
