from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from utils import get_meta_and_labels, label_file_to_image

from encord_active.lib.labels.label_transformer import (
    DataLabel,
    LabelTransformer,
    PolygonLabel,
)


class PolyTransformer(LabelTransformer):
    def from_custom_labels(
        self, label_files: List[Path], data_files: List[Path]
        """This function takes in a list of label files and a list of data files and returns a list of DataLabel objects. It uses the label files to extract metadata and labels, and then processes each label file to create a DataLabel object. The function uses the label files to extract metadata and labels, and then processes each label file to create a DataLabel object. The function also uses the data files to extract image data and normalize it for processing. The function then iterates through each instance in the image and creates a DataLabel object with the corresponding class and polygon label. The function returns a list of all the created DataLabel objects.
        Parameters:
            - label_files (List[Path]): A list of label files to extract metadata and labels from.
            - data_files (List[Path]): A list of data files to extract image data from.
        Returns:
            - out (List[DataLabel]): A list of DataLabel objects containing the extracted labels and image data.
        Processing Logic:
            - Extracts metadata and labels from label files.
            - Extracts image data from data files.
            - Normalizes image data for processing.
            - Iterates through each instance in the image.
            - Creates a DataLabel object with the corresponding class and polygon label.
            - Returns a list of all the created DataLabel objects."""
        
    ) -> List[DataLabel]:
        meta, label_files = get_meta_and_labels(label_files, extension=".png")

        out = []
        for label_file in label_files:
            classes = meta[label_file.parent.name]["objects"]
            image_file = label_file_to_image(label_file)

            image = np.asarray(Image.open(label_file))

            h, w = image.shape[:2]
            normalization = np.array([[w, h]], dtype=float)

            for instance_id in np.unique(image):
                if instance_id == 0:
                    continue

                instance_mask = (image == instance_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    contour = contour.squeeze() / normalization

                    out.append(
                        DataLabel(
                            abs_data_path=image_file,
                            label=PolygonLabel(
                                class_=classes.get(str(instance_id), {}).get(
                                    "category", "unknown"
                                ),
                                polygon=contour,
                            ),
                        )
                    )
                    break

        return out
