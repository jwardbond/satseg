import numpy as np


def deep_features(
    image_tensor,
    extractor,
    layer,
    facet,
    bin: bool = False,
    include_cls: bool = False,
    device="cuda",
):
    """Extract descriptors from pretrained DINO model; Create an adj matrix from descriptors

    Args:
        image_tensor (torch.Tensor): Tensor of size BxCxHxW
        extractor (DINO): Initialized model to extract descriptors from
        layer (str): Layer to extract the descriptors from
        facet (str): Facet to extract the descriptors from (key, value, query)
        bin (bool, optional): apply log binning to the descriptor. default is False.
        include_cls (bool, optional): To include CLS token in extracted descriptor
        device (str, optional): Training device. Defaults to 'cuda'.

    Returns:
        np.ndarray: (N*B)xD, usually just NxD
    """

    # images to deep_features.
    # input is size: BxCxHxW
    # output is size: Bx1xNxD
    deep_features = (
        extractor.extract_descriptors(
            image_tensor.to(device), layer, facet, bin, include_cls
        )
        .cpu()
        .numpy()
    )

    # in: Bx1xNxD
    # out: BxNxD
    deep_features = np.squeeze(deep_features, axis=1)

    # in: BxNxD
    # out: (B*N)xD
    deep_features = deep_features.reshape(
        (deep_features.shape[0] * deep_features.shape[1], deep_features.shape[2])
    )

    return deep_features
