import os
import argparse
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

##########################################################################################
# Adapted from https://github.com/SAMPL-Weizmann/DeepCut
##########################################################################################


def GNN_seg_dataset(
    mode,
    cut,
    alpha,
    epochs,
    K,
    pretrained_weights,
    in_dir,
    out_dir,
    save,
    cc,
    bs,
    log_bin,
    res,
    facet,
    layer,
    stride,
    device,
):
    """Segment images in a dataset using ViT+GNN methodology

    Get bounding box (k==2 only) or segmentation maps bounding boxes
    will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')

    Args:
        cut: chosen clustering functional: NCut==1, CC==0
        epochs: Number of epochs for every step in image
        K: Number of segments to search in each image
        pretrained_weights: Weights of pretrained images
        dir: Directory for chosen dataset
        out_dir: Output directory to save results
        cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
        b_box: If true will output bounding box (for k==2 only), else segmentation map
        log_bin: Apply log binning to the descriptors (correspond to smother image)
        device: Device to use ('cuda'/'cpu')
    """
    ##########################################################################################
    # Dino model init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )
    # VIT small feature dimension, with or without log bin
    if not log_bin:
        feats_dim = 384
    else:
        feats_dim = 6528

    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    ##########################################################################################
    # GNN model init
    ##########################################################################################
    # import cutting gnn model if cut == 0 NCut else CC
    if cut == 0:
        from satseg.gnn_pool import GNNpool
    else:
        from satseg.gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), "model.pt")
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        torch.save(model2.state_dict(), "model2.pt")
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        torch.save(model3.state_dict(), "model3.pt")
        model3.train()

    ##########################################################################################
    # Iterate over files in input directory and apply GNN segmentation
    ##########################################################################################
    for filename in tqdm(os.listdir(in_dir)):
        # If not image, skip
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue
        # if file already processed
        if os.path.exists(os.path.join(out_dir, filename.split(".")[0] + ".txt")):
            continue
        if os.path.exists(os.path.join(out_dir, filename)):
            continue

        ##########################################################################################
        # Data loading
        ##########################################################################################
        # loading images
        image_tensor, image = utils.load_data_img(os.path.join(in_dir, filename), res)
        # Extract deep features, from the transformer and create an adj matrix
        F = deep_features(
            image_tensor, extractor, layer, facet, bin=log_bin, device=device
        )
        W = utils.create_adj(F, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = utils.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(
            torch.load("./model.pt", map_location=torch.device(device))
        )
        opt = optim.AdamW(model.parameters(), lr=0.001)

        ##########################################################################################
        # GNN pass
        ##########################################################################################
        for _ in range(epochs[0]):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()

        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)

        ##########################################################################################
        # Post-processing Connected Component/bilateral solver
        ##########################################################################################
        mask0, S = utils.graph_to_mask(S, cc, stride, image_tensor, image)
        # apply bilateral solver
        if bs:
            mask0 = bilateral_solver_output(image, mask0)[1]

        if mode == 0:
            utils.save_or_show(
                [image, mask0, utils.apply_seg_map(image, mask0, 0.7)],
                filename,
                out_dir,
                save,
            )
            continue

        ##########################################################################################
        # Second pass on foreground
        ##########################################################################################
        # extracting foreground sub-graph
        sec_index = np.nonzero(S).squeeze(1)
        F_2 = F[sec_index]
        W_2 = utils.create_adj(F_2, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = utils.load_data(W_2, F_2)
        data_2 = Data(node_feats, edge_index, edge_weight).to(device)
        # re-init weights and optimizer for every image
        model2.load_state_dict(
            torch.load("./model2.pt", map_location=torch.device(device))
        )
        opt = optim.AdamW(model2.parameters(), lr=0.001)

        ####################################################
        # GNN pass
        ####################################################
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

        mask1, S = utils.graph_to_mask(S, cc, stride, image_tensor, image)

        if mode == 1:
            utils.save_or_show(
                [image, mask1, utils.apply_seg_map(image, mask1, 0.7)],
                filename,
                out_dir,
                save,
            )
            continue

        ##########################################################################################
        # Second pass background
        ##########################################################################################
        # extracting background sub-graph
        sec_index = np.nonzero(S == 0).squeeze(1)
        F_3 = F[sec_index]
        W_3 = utils.create_adj(F_3, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = utils.load_data(W_3, F_3)
        data_3 = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model3.load_state_dict(
            torch.load("./model3.pt", map_location=torch.device(device))
        )
        opt = optim.AdamW(model3.parameters(), lr=0.001)
        for _ in range(epochs[2]):
            opt.zero_grad()
            A_3, S_3 = model3(data_3, torch.from_numpy(W_3).to(device))
            loss = model3.loss(A_3, S_3)
            loss.backward()
            opt.step()

        # fusing subgraph and original graph
        S_3 = S_3.detach().cpu()
        S_3 = torch.argmax(S_3, dim=-1)
        S[sec_index] = S_3 + foreground_k + 5

        mask2, S = utils.graph_to_mask(S, cc, stride, image_tensor, image)
        if bs:
            mask_foreground = mask0
            mask_background = np.where(mask2 != foreground_k + 5, 0, 1)
            bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
            bs_background = bilateral_solver_output(image, mask_background)[1]
            mask2 = bs_foreground + (bs_background * 2)

        utils.save_or_show(
            [image, mask2, utils.apply_seg_map(image, mask2, 0.7)],
            filename,
            out_dir,
            save,
        )


def GNN_seg_image(
    mode: int,
    device: str,
    in_dir: str,
    out_dir: str,
    save: bool,
    pretrained_weights: str,
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

    ##########################################################################################
    # ViT init
    ##########################################################################################
    extractor = ViTExtractor(
        "dino_vits8", stride, model_dir=pretrained_weights, device=device
    )


def process_config(confpath):
    """Loads and processes the config file for running satseg"""
    with open(confpath) as f:
        config = yaml.safe_load(f)

    config = defaultdict(None, config)

    config["device"] = config["device"] if torch.cuda.is_available() else "cpu"
    config["res"] = tuple(config["res"])

    assert not (
        (config["K"] != 2) and (config["cc"] is not None)
    ), "largest connected component only available for k == 2"

    if config["cut"] == 1:  # If Correlational Clustering, set max # clusters
        config["K"] = 10

    # If Directory doesn't exist than download
    if not os.path.exists(config["pretrained_weights"]):
        url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth"
        utils.download_url(url, config["pretrained_weights"])

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="satseg",
        description="Segments images using ViT + GNN according to https://arxiv.org/pdf/2212.05853.pdf",
    )

    parser.add_argument("confpath")
    parser.add_argument("-f", "--filepath", help="specify image to segment")
    args = parser.parse_args()

    config = process_config(args.confpath)

    if args.filepath:
        GNN_seg_image(**config)
    else:
        GNN_seg_dataset(**config)
