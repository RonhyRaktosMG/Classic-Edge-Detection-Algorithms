from abc import ABC, abstractmethod
import numpy as np


class EdgeDetector(ABC):

    @abstractmethod
    def detect(
        self,
        grayscale: np.ndarray,
        threshold: float | None = None
    ):
        pass