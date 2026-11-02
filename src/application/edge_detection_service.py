from src.core.preprocessing.grayscale import to_grayscale


class EdgeDetectionService:

    def __init__(self, reader, writer, detector):

        self.reader = reader
        self.writer = writer
        self.detector = detector

    def execute(
        self,
        input_path,
        output_path,
        threshold=None
    ):

        # Lecture
        image = self.reader.read(input_path)

        # Prétraitement
        grayscale = to_grayscale(image)

        # Détection
        result = self.detector.detect(
            grayscale,
            threshold
        )

        # Sauvegarde
        self.writer.write(
            result.edges,
            output_path
        )

        return result