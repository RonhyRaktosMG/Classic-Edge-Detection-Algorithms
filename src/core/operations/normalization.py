import numpy as np


def normalize_to_uint8(image):

    image = np.asarray(image, dtype=np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (
        (image - minimum)
        * (255.0 / (maximum - minimum))
    )

    return np.clip(normalized, 0, 255).astype(np.uint8)