from pathlib import Path
import numpy as np
from PIL import Image


class PillowImageReader:

    def read(self, path):

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            return np.asarray(
                image.convert("RGB"),
                dtype=np.uint8
            ).copy()