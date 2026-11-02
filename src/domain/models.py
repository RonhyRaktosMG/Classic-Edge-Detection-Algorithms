from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EdgeDetectionResult:

    edges: np.ndarray

    gradient_x: np.ndarray | None = None
    gradient_y: np.ndarray | None = None
    magnitude: np.ndarray | None = None