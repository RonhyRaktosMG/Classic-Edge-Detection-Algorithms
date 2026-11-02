from pathlib import Path
import numpy as np
from PIL import Image


class PillowImageWriter:

    def write(self, image, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        image = np.clip(
            image, 0, 255
        ).astype(np.uint8)

        Image.fromarray(image).save(path)