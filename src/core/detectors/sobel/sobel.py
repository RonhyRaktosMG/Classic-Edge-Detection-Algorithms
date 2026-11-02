import numpy as np

from src.domain.interfaces import EdgeDetector
from src.domain.models import EdgeDetectionResult

from src.core.operations.convolution import convolve2d
from src.core.operations.gradient import gradient_magnitude
from src.core.operations.normalization import normalize_to_uint8
from src.core.operations.threshold import apply_threshold


class SobelDetector(EdgeDetector):

    SOBEL_X = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    SOBEL_Y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    def detect(self, grayscale, threshold=None):

        image = np.asarray(grayscale, dtype=np.float32)

        if image.ndim != 2 or image.size == 0:
            raise ValueError("Image en niveaux de gris non vide attendue.")

        # Calcul des gradients
        gx = convolve2d(image, self.SOBEL_X)

        gy = convolve2d(image, self.SOBEL_Y)

        # Calcul de la magnitude
        magnitude = gradient_magnitude(gx, gy)

        # Normalisation
        normalized = normalize_to_uint8(magnitude)

        # Seuillage facultatif
        if threshold is not None:
            edges = apply_threshold(normalized, threshold)
        else:
            edges = normalized

        return EdgeDetectionResult(
            edges=edges,
            gradient_x=gx,
            gradient_y=gy,
            magnitude=magnitude
        )