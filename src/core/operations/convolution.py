import numpy as np


def convolve2d(image, kernel):

    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    kh, kw = kernel.shape

    pad_y = kh // 2
    pad_x = kw // 2

    padded = np.pad(
        image,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="edge"
    )

    output = np.zeros(image.shape, dtype=np.float32)

    flipped_kernel = np.flip(kernel, axis=(0, 1))

    height, width = image.shape

    for y in range(height):
        for x in range(width):

            region = padded[y:y + kh, x:x + kw]

            output[y, x] = np.sum(
                region * flipped_kernel
            )

    return output