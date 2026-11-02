import numpy as np


def to_grayscale(image):

    image = np.asarray(image)

    if image.ndim == 2:
        return image.astype(np.float32)

    rgb = image[..., :3].astype(np.float32)

    grayscale = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )

    return grayscale.astype(np.float32)