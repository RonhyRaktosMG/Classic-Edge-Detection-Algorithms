import numpy as np


def apply_threshold(image, threshold):

    return np.where(
        image >= threshold,
        255,
        0
    ).astype(np.uint8)