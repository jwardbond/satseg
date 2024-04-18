import argparse
import pathlib
from collections import defaultdict

import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def get_correspondence_map(
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

    ##########################################################################################
    # Visualize
    ##########################################################################################
    # Get coordinates of patch in original image
    original_shape = (source.shape[0], source.shape[1])
    scaled_shape = (
        int(1 + (target_tensor.shape[2] - P) // stride),
        int(1 + (target_tensor.shape[3] - P) // stride),
    )  # Image as patches

    patch_coords = np.unravel_index(source_patch, scaled_shape)
    scale = np.divide(original_shape, scaled_shape)
    scaled_coords = np.multiply(patch_coords, scale)

    # Save or show
    if save:
        # Source image + annotation
        annulus = patches.Annulus(
            (scaled_coords[1], scaled_coords[0]), r=120, width=30, color="r"
        )
        plt.imshow(source, cmap="magma")
        ax = plt.gca()
        ax.add_patch(annulus)
        ax.set_axis_off()
        plt.savefig(out_dir / "source.png", dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Target image
        plt.imsave(
            out_dir / "target.png",
            dpi=200,
            arr=target,
        )

        # Heat map
        output = cv2.resize(
            output.astype("float"),
            (target[:, :, 0].shape[1], target[:, :, 0].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        plt.imsave(
            out_dir / f"output_{facet}{layer}.png",
            dpi=200,
            arr=output,
            cmap="magma",
        )

    else:
        correspondence_plot(
            [source, target, output],
            "Source, Target, Correspondence",
            patch_coords,
            scaled_coords,
        )


def correspondence_plot(
    im_arr: list,
    title: str,
    patch_coords: tuple[int, int],
    scaled_coords: tuple[int, int],
):
    """Display images N in a row from arbitrary number of images in a list

    Adapted from https://github.com/SAMPL-Weizmann/DeepCut/blob/main/util.py

    Args:
        im_arr: array of images as numpy arrays in [source, target, output]
        title: Window name
        coords: coordinates of the source patch
    """
    fig, axes = plt.subplots(
        1,
        3,
        squeeze=False,
        dpi=200,
    )

    org = patches.Annulus((patch_coords[1], patch_coords[0]), r=4, width=1, color="r")

    annulus = patches.Annulus(
        (scaled_coords[1], scaled_coords[0]), r=120, width=30, color="r"
    )

    axes[0, 2].add_patch(org)
    axes[0, 0].add_patch(annulus)

    count = 0
    for i in range(len(im_arr)):
        axes[count // 3][count % 3].imshow(im_arr[i], cmap="magma")
        axes[count // 3][count % 3].axis("off")
        count = count + 1
    # Delete axis for non-full rows
    for i in range(len(im_arr) + 1, 3):
        axes[count // 3][count % 3].axis("off")
        count = count + 1

    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
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
        prog="Correspondence",
        description="Computes similiarity between a given patch in a source image, and all patches in a target image",
    )
    parser.add_argument("confpath")
    parser.add_argument("sourcepath", help="Path to source image")
    parser.add_argument("targetpath", help="Path to target image")
    parser.add_argument("patchnum", help="Patch number for source image")

    args = parser.parse_args()
    config, raw_config = process_config(args.confpath)

    sourcepath = pathlib.Path(args.sourcepath)
    targetpath = pathlib.Path(args.targetpath)
    patchnum = int(args.patchnum)

    if config["save"]:
        # Create output dir
        filename = f"{sourcepath.stem}_{targetpath.stem}_{patchnum}"
        config["out_dir"] = config["out_dir"] / filename
        config["out_dir"].mkdir(parents=True, exist_ok=True)

        # # Save raw config file
        # with open(config["out_dir"] / "config.yml", "w") as f:
        #     yaml.dump(raw_config, f, sort_keys=False)

    # Run
    get_correspondence_map(
        **config,
        source_image=sourcepath,
        target_image=targetpath,
        source_patch=patchnum,
    )
