import argparse
import pathlib
from collections import defaultdict

import yaml
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch_geometric.data import Data

import utils
from satseg.bilateral_solver import bilateral_solver_output
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


def GNN_seg_image(
    mode: int,
    device: str,
    out_dir: pathlib.PurePath,
    save: bool,
    pretrained_weights: pathlib.PurePath,
    res: tuple[int, int],
    stride: int,
    layer: int,
    facet: str,
    cut: int,
    alpha: int,
    K: int,
    epochs: list[int],
    cc: bool,
    bs: bool,
    log_bin: bool,
    imgpath: pathlib.PurePath,
    **kwargs,
):
    """Segment images in a dataset using ViT+GNN methodology

    Get bounding box (k==2 only) or segmentation maps bounding boxes
    will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')

    Args:
        mode: mode to run the segmentation in: normal==0, 2-stage on foreground == 1, 2-stage on fore+background == 2
        device: Device to use ('cuda'/'cpu')
        in_dir: Filepath of input image
        out_dir: Filepath for results
        save: True to save results, else false
        cut: chosen clustering functional: NCut==1, CC==0
        alpha: k-sensitivity param
        pretrained_weights: Path to pretrained ViT
        res: Resolution of image
        stride: Stride for feature extraction
        layer: Layer of ViT to extract features from
        facet: Key, Query, or Value for feature extraction
        epochs: list of # epochs for each mode
        K: Number of segments to generate in the image
        cc: True to show only the largest component of clustering (K==2)
        bs: True to use bilateral solver during post-processing
        log_bin: True to use log binning during post-processing
    """

    # Set up tmp directory
    tmp_dir = pathlib.Path("./.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ##########################################################################################
    # ViT init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )

    if not log_bin:
        feats_dim = 384  # Default for ViT-small DiNO
    else:
        feats_dim = 6528

    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    ##########################################################################################
    # GNN init
    ##########################################################################################
    if cut == 0:
        from satseg.gnn_pool import GNNpool
    else:
        from satseg.gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        model3.train()

    opt = optim.AdamW(model.parameters(), lr=0.001)

    ##########################################################################################
    # Load Data
    ##########################################################################################
    image_tensor, image = utils.load_data_img(imgpath, res)

    F = deep_features(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
    W = utils.create_adj(F, cut, alpha)
    # pd.DataFrame(W).to_csv("adj.csv")

    # Data to pytorch_geometric format
    node_feats, edge_index, edge_weight = utils.load_data(W, F)
    data = Data(node_feats, edge_index, edge_weight).to(device)

    ##########################################################################################
    # GNN pass
    ##########################################################################################
    losses = []
    epchs = []
    for i in tqdm(range(epochs[0]), ncols=50):
        opt.zero_grad()
        A, S = model(data, torch.from_numpy(W).to(device))
        loss = model.loss(A, S)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        epchs.append(i)

    S = S.detach().cpu()
    S = torch.argmax(S, dim=-1)  # selects class with highest probability

    if save:
        utils.save_loss(
            losses,
            epchs,
            out_dir / (imgpath.stem + "_losses.csv"),
        )

    ##########################################################################################
    # Post-processing
    ##########################################################################################
    P = extractor.model.patch_embed.patch_size
    mask0, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)

    if bs:
        mask0 = bilateral_solver_output(image, mask0)[1]

    if mode == 0:
        utils.save_or_show(
            [image, mask0, utils.apply_seg_map(image, mask0, 0.7)],
            imgpath.stem,
            out_dir,
            save,
        )
        return

    ##########################################################################################
    # Second pass on foreground
    ##########################################################################################
    sec_index = np.nonzero(S).squeeze(1)

    F_2 = F[sec_index]
    W_2 = utils.create_adj(F_2, cut, alpha)

    # Data to pytorch_geometric format
    node_feats, edge_index, edge_weight = utils.load_data(W_2, F_2)
    data_2 = Data(node_feats, edge_index, edge_weight).to(device)

    # GNN Pass
    opt = optim.AdamW(model2.parameters(), lr=0.001)
    for _ in tqdm(range(epochs[1]), ncols=50):
        opt.zero_grad()
        A_2, S_2 = model2(data_2, torch.from_numpy(W_2).to(device))
        loss = model2.loss(A_2, S_2)
        loss.backward()
        opt.step()

    # fusing subgraph and original graph
    S_2 = S_2.detach().cpu()
    S_2 = torch.argmax(S_2, dim=-1)
    S[sec_index] = S_2 + 3

    mask2, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)

    if mode == 1:
        utils.save_or_show(
            [image, mask2, utils.apply_seg_map(image, mask2, 0.7)],
            imgpath.stem,
            out_dir,
            save,
        )
        return

    ##########################################################################################
    # Second pass on background
    ##########################################################################################
    sec_index = np.nonzero(S == 0).squeeze(1)
    F_3 = F[sec_index]
    W_3 = utils.create_adj(F_3, cut, alpha)

    node_feats, edge_index, edge_weight = utils.load_data(W_3, F_3)
    data_3 = Data(node_feats, edge_index, edge_weight).to(device)

    # GNN Pass
    opt = optim.AdamW(model3.parameters(), lr=0.001)
    for _ in tqdm(range(epochs[2]), ncols=50):
        opt.zero_grad()
        A_3, S_3 = model3(data_3, torch.from_numpy(W_3).to(device))
        loss = model3.loss(A_3, S_3)
        loss.backward()
        opt.step()

    # Fusing
    S_3 = S_3.detach().cpu()
    S_3 = torch.argmax(S_3, dim=-1)
    S[sec_index] = S_3 + foreground_k + 5

    mask3, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)
    if bs:
        mask_foreground = mask0
        mask_background = np.where(mask3 != foreground_k + 5, 0, 1)
        bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
        bs_background = bilateral_solver_output(image, mask_background)[1]
        mask3 = bs_foreground + (bs_background * 2)

    utils.save_or_show(
        [image, mask3, utils.apply_seg_map(image, mask3, 0.7)],
        imgpath.stem,
        out_dir,
        save,
    )


def GNN_seg_dataset(
    mode: int,
    device: str,
    in_dir: pathlib.PurePath,
    out_dir: pathlib.PurePath,
    save: bool,
    pretrained_weights: pathlib.PurePath,
    res: tuple[int, int],
    stride: int,
    layer: int,
    facet: str,
    cut: int,
    alpha: int,
    K: int,
    epochs: list[int],
    cc: bool,
    bs: bool,
    log_bin: bool,
    filename_append: str = "",
    **kwargs,
):
    """Segment images in a dataset using ViT+GNN methodology

    Get bounding box (k==2 only) or segmentation maps bounding boxes
    will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')

    Args:
        mode: mode to run the segmentation in: normal==0, 2-stage on foreground == 1, 2-stage on fore+background == 2
        device: Device to use ('cuda'/'cpu')
        in_dir: Filepath of input image
        out_dir: Filepath for results
        save: True to save results, else false
        cut: chosen clustering functional: NCut==1, CC==0
        alpha: k-sensitivity param
        pretrained_weights: Path to pretrained ViT
        res: Resolution of image
        stride: Stride for feature extraction
        layer: Layer of ViT to extract features from
        facet: Key, Query, or Value for feature extraction
        epochs: list of # epochs for each mode
        K: Number of segments to generate in the image
        cc: True to show only the largest component of clustering (K==2)
        bs: True to use bilateral solver during post-processing
        log_bin: True to use log binning during post-processing
        filename_append: Appended string to output filename
    """

    tmp_dir = pathlib.Path("./.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ##########################################################################################
    # ViT init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )

    if not log_bin:
        feats_dim = 384  # Default for ViT-small DiNO
    else:
        feats_dim = 6528

    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    ##########################################################################################
    # GNN init
    ##########################################################################################
    if cut == 0:
        from satseg.gnn_pool import GNNpool
    else:
        from satseg.gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), tmp_dir / "model.pt")
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        torch.save(model2.state_dict(), tmp_dir / "model2.pt")
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        torch.save(model3.state_dict(), tmp_dir / "model3.pt")
        model3.train()

    ##########################################################################################
    # Load Data
    ##########################################################################################
    for imgpath in tqdm(
        list(in_dir.glob("*.jpg"))
        + list(in_dir.glob("*.png"))
        + list(in_dir.glob("*.jpeg")),
        ncols=100,
    ):
        tqdm.write(f"Filename {imgpath}")
        image_tensor, image = utils.load_data_img(imgpath, res)

        F = deep_features(
            image_tensor, extractor, layer, facet, bin=log_bin, device=device
        )
        W = utils.create_adj(F, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = utils.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        ##########################################################################################
        # GNN pass
        ##########################################################################################

        # Load starting weights
        model.load_state_dict(
            torch.load(tmp_dir / "model.pt", map_location=torch.device(device))
        )
        opt = optim.AdamW(model.parameters(), lr=0.001)

        losses = []
        epchs = []
        for i in range(epochs[0]):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()

            # Logging
            losses.append(loss.item())
            epchs.append(i)

        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)  # selects class with highest probability

        if save:
            utils.save_loss(
                losses,
                epchs,
                out_dir / (imgpath.stem + filename_append + "_losses.csv"),
            )

        ##########################################################################################
        # Post-processing
        ##########################################################################################
        P = extractor.model.patch_embed.patch_size
        mask0, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)

        if bs:
            mask0 = bilateral_solver_output(image, mask0)[1]

        if mode == 0:
            utils.save_or_show(
                [image, mask0, utils.apply_seg_map(image, mask0, 0.7)],
                imgpath.stem + filename_append,
                out_dir,
                save,
            )
            continue

        ##########################################################################################
        # Second pass on foreground
        ##########################################################################################
        sec_index = np.nonzero(S).squeeze(1)
        F_2 = F[sec_index]
        W_2 = utils.create_adj(F_2, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = utils.load_data(W_2, F_2)
        data_2 = Data(node_feats, edge_index, edge_weight).to(device)

        # GNN Pass
        model2.load_state_dict(
            torch.load(tmp_dir / "model2.pt", map_location=torch.device(device))
        )

        opt = optim.AdamW(model2.parameters(), lr=0.001)
        for _ in range(epochs[1]):
            opt.zero_grad()
            A_2, S_2 = model2(data_2, torch.from_numpy(W_2).to(device))
            loss = model2.loss(A_2, S_2)
            loss.backward()
            opt.step()

        # fusing subgraph and original graph
        S_2 = S_2.detach().cpu()
        S_2 = torch.argmax(S_2, dim=-1)
        S[sec_index] = S_2 + 3

        mask2, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)

        if mode == 1:
            utils.save_or_show(
                [image, mask2, utils.apply_seg_map(image, mask2, 0.7)],
                imgpath.stem + filename_append,
                out_dir,
                save,
            )
            continue

        ##########################################################################################
        # Second pass on background
        ##########################################################################################
        sec_index = np.nonzero(S == 0).squeeze(1)
        F_3 = F[sec_index]
        W_3 = utils.create_adj(F_3, cut, alpha)

        node_feats, edge_index, edge_weight = utils.load_data(W_3, F_3)
        data_3 = Data(node_feats, edge_index, edge_weight).to(device)

        # GNN Pass
        model3.load_state_dict(
            torch.load(tmp_dir / "model3.pt", map_location=torch.device(device))
        )

        opt = optim.AdamW(model3.parameters(), lr=0.001)
        for _ in range(epochs[2]):
            opt.zero_grad()
            A_3, S_3 = model3(data_3, torch.from_numpy(W_3).to(device))
            loss = model3.loss(A_3, S_3)
            loss.backward()
            opt.step()

        # Fusing
        S_3 = S_3.detach().cpu()
        S_3 = torch.argmax(S_3, dim=-1)
        S[sec_index] = S_3 + foreground_k + 5

        mask3, S = utils.graph_to_mask(S, cc, stride, P, image_tensor, image)
        if bs:
            mask_foreground = mask0
            mask_background = np.where(mask3 != foreground_k + 5, 0, 1)
            bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
            bs_background = bilateral_solver_output(image, mask_background)[1]
            mask3 = bs_foreground + (bs_background * 2)

        utils.save_or_show(
            [image, mask3, utils.apply_seg_map(image, mask3, 0.7)],
            imgpath.stem + filename_append,
            out_dir,
            save,
        )


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
        prog="satseg",
        description="Segments images using ViT + GNN according to https://arxiv.org/pdf/2212.05853.pdf",
    )
    parser.add_argument("confpath")
    parser.add_argument("-f", "--filepath", help="specify image to segment")
    parser.add_argument(
        "-e",
        "--experiment",
        help="run as experiment",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    config, raw_config = process_config(args.confpath)

    if args.filepath:
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
        GNN_seg_image(**config, imgpath=imgpath)

    elif args.experiment:
        if config["save"]:
            # Create specific output dir
            config["out_dir"] = config["out_dir"].mkdir(parents=True, exist_ok=True)

            # Save raw config file
            with open(config["out_dir"] / "config.yml", "w") as f:
                yaml.dump(raw_config, f, sort_keys=False)

        GNN_seg_dataset(**config)
