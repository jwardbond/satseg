from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit
from tqdm import tqdm
from PIL import Image
import urllib.request
import numpy as np
import pathlib
import torch
import cv2

cmap = "tab20"


def save_loss(loss: list[float], epochs: list[int], outpath: pathlib.PurePath):
    """Save or show loss plot

    Args:
        loss: List of loss values
        epochs: List of epochs
        dir: Directory to save the plot
    """
    df = pd.DataFrame({"epochs": epochs, "loss": loss})
    df.to_csv(outpath, index=False)


def save_or_show(arr, filename: str, dir: pathlib.PurePath, save=False):
    if save:
        plt.imsave(dir / (filename + "_org" + ".png"), arr[0], cmap=cmap)
        plt.imsave(dir / (filename + "_mask" + ".png"), arr[1], cmap=cmap)
        plt.imsave(dir / (filename + "_fused" + ".png"), arr[2], cmap=cmap)
    else:
        im_show_n(arr, 3, "org, mask, fused")


def graph_to_mask(
    S: torch.Tensor,
    cc: bool,
    stride: int,
    patch_size: int,
    image_tensor: torch.Tensor,
    image: np.ndarray,
):
    """
    Args:
        S: Segmentation map (Nx1)
        cc: Connected component flag
        stride: Stride length used to generate patches
        patch_size: Size of patches (P)
        image_tensor: Tensor of original image after scaling (1xCxHxW)
        imageL: Original image as a numpy array (HxWxC)
    Returns:
        mask: Segmentation mask as a numpy array (HxW)
        S: Segmentation map as a tensor
    """
    # Reshape clustered graph to size of image
    S = np.array(
        torch.reshape(
            S,
            (
                int(
                    1 + (image_tensor.shape[2] - patch_size) // stride
                ),  # height in # patches
                int(
                    1 + (image_tensor.shape[3] - patch_size) // stride
                ),  # width in # patches
            ),
        )
    )  # HxW (in patches)

    ## check if background is 0 and main object is 1 in segmentation map
    ## checks each corner
    ## Commented out for satellite images (no "background")
    # inverts if not
    # if (
    #     S[0][0]
    #     + S[S.shape[0] - 1][0]
    #     + S[0][S.shape[1] - 1]
    #     + S[S.shape[0] - 1][S.shape[1] - 1]
    # ) > 2:
    #     S = 1 - S

    # chose largest component (for k == 2)
    if cc:
        S = largest_cc(S)

    # mask to original image size
    mask = cv2.resize(
        S.astype("float"),
        (image[:, :, 0].shape[1], image[:, :, 0].shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    S = torch.tensor(np.reshape(S, (S.shape[0] * S.shape[1],))).type(torch.LongTensor)

    return mask, S


def create_adj(F, cut, alpha=1):
    """Create adjacency matrix from feature vectors

    Args:
        F: Feature matrix (NxD)
        cut: Cut type (0 for NCut, 1 for CC)
        alpha: Alpha value for CC cut

    Returns:
        W: Adjacency matrix (NxN)
    """
    W = F @ F.T

    # if NCut
    if cut == 0:
        # threshold
        W = W * (W > 0)
        # norm
        W = W / W.max()

    # if CC
    else:
        W = W - (W.max() / alpha)

    return W


def im_show_n(im_arr: list, n: int, title: str):
    """Display images N in a row from arbitrary number of images in a list

    Args:
        im_arr: array of images
        n: Number of subplots in a row
        title: Window name

    @author:Amit
    """
    fig, axes = plt.subplots(
        len(im_arr) // n if len(im_arr) % n == 0 else len(im_arr) // n + 1,
        n,
        squeeze=False,
        dpi=200,
    )

    count = 0
    for i in range(len(im_arr)):
        axes[count // n][count % n].imshow(im_arr[i])
        axes[count // n][count % n].axis("off")
        count = count + 1
    # Delete axis for non-full rows
    for i in range(len(im_arr) + 1, n):
        axes[count // n][count % n].axis("off")
        count = count + 1

    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()


@njit()
def discr_ncut(A, B, deg, W):
    """
    Calculate discrete normalized-cut of a given graph for k=2 cut.
    @param A: First cluster of nodes
    @param B: Second cluster of nodes
    @param deg: Array of node degrees
    @param W: Adjacency matrix
    @return: Normalized-cut value
    """
    # sum of cut edges
    cut_size = 0
    for i in range(A[0].shape[0]):
        for j in range(B[0].shape[0]):
            cut_size = cut_size + W[A[0][i]][B[0][j]]
    # sum of out degrees
    ncut = 1.0 / np.sum(deg[A[0]]) + 1.0 / np.sum(deg[B[0]])
    ncut = cut_size * ncut

    return ncut


# suggested use of discr_ncut
"""
from torch_geometric.utils import degree

sum of cut edges
deg = degree(edge_index[0])
A = np.where(S == 0)
B = np.where(S == 1)
ncut = discr_ncut(A, B, np.array(deg), W)
"""


def load_data(adj: np.ndarray, node_feats: np.ndarray):
    """Load data to pytorch-geometric data format

    Args:
        adj: Adjacency matrix of a graph (NxN)
        node_feats: Feature matrix of a graph (NxD)

    Returns:
        node_feats: tensor of node features (NxD)
        edge_index: tensor of edge indices (2xE)
        edge_weight: tensor of edge weights (Ex1)
    """
    node_feats = torch.from_numpy(node_feats)
    edge_index = torch.from_numpy(np.array(np.nonzero(adj > 0)))
    row, col = edge_index
    edge_weight = torch.from_numpy(adj[row, col])

    return node_feats, edge_index, edge_weight


def load_data_img(imgpth: str, image_size: int):
    """Loads an image from a file

    Args:
        imgpth (str): Path to the image file
        image_size (int): Desired # of pixels for smallest edge in image

    Returns:
        image_tensor (torch.Tensor): Resized image as a tensor (1 x C x H_new x W_new)
        image (np.ndarray): Original image as a numpy array (HxWxC)
    """

    # Load image
    pil_image = Image.open(imgpth).convert("RGB")

    # Define transformations
    prep = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Resized image tensor
    image_tensor = prep(pil_image)[None, ...]

    # To numpy array
    image = np.array(pil_image)

    return image_tensor, image


def largest_cc(S: np.ndarray):
    """Gets a segmentation map and finds the largest connected component, discards the rest of the segmentation map.

    Args:
        S: Segmentation map

    Returns:
        Largest connected component in given segmentation map
    """
    us_cc = cv2.connectedComponentsWithStats(S.astype("uint8"), connectivity=4)
    # get indexes of sorted sizes for CCs
    us_cc_stat = us_cc[2]
    cc_idc = np.argsort(us_cc_stat[:, -1])[::-1]
    # decision rule for crop
    if np.percentile(S[us_cc[1] == cc_idc[0]], 99) == 0:
        # 99th percentile of biggest connected component is 0 -> cc_idc[0] is background
        mask: np.ndarray = np.equal(us_cc[1], cc_idc[1])
    elif np.percentile(S[us_cc[1] == cc_idc[1]], 99) == 0:
        # 99th percentile of 2nd biggest connected component is 0 -> cc_idc[0] is background
        mask: np.ndarray = np.equal(us_cc[1], cc_idc[0])
    else:
        raise NotImplementedError("No valid decision rule for cropping")

    return mask


def apply_seg_map(img, seg, alpha):
    """
    Overlay segmentation map onto an image, the function is jited for performance.
    @param img: input image as numpy array
    @param seg: input segmentation map as a numpy array
    @param alpha: The opacity of the segmentation overlay, 0==transparent, 1==only segmentation map
    @return: segmented image as a numpy array
    """
    tmp_path = pathlib.Path("./.tmp/")
    tmp_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(tmp_path / "tmp.png", seg, cmap=cmap)
    seg = (plt.imread(tmp_path / "tmp.png")[:, :, :3] * 255).astype(np.uint8)
    return ((seg * alpha) + (img * (1 - alpha))).astype(np.uint8)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
