from src.application.edge_detection_service import EdgeDetectionService

from src.core.detectors.sobel import SobelDetector

from src.infrastructure.image_reader import PillowImageReader
from src.infrastructure.image_writer import PillowImageWriter


def main():

    service = EdgeDetectionService(
        reader=PillowImageReader(),
        writer=PillowImageWriter(),
        detector=SobelDetector()
    )

    result = service.execute(
        input_path="images/input.png",
        output_path="output/sobel.png",
        threshold=None
    )

    print("Détection terminée !")
    print("Dimensions :", result.edges.shape)


if __name__ == "__main__":
    main()