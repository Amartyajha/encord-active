import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from encord_active.lib.labels.label_transformer import (
    DataLabel,
    LabelTransformer,
    PolygonLabel,
)


def get_meta_and_labels(label_files: List[Path], extension: str = ".json"):
    """"Get metadata and labels from a list of label files.
    Parameters:
        - label_files (List[Path]): List of label files to process.
        - extension (str): Optional. File extension of label files. Defaults to ".json".
    Returns:
        - meta (dict): Dictionary containing metadata for videos.
        - label_files (List[Path]): List of label files without the meta file.
    Processing Logic:
        - Find the meta file in the list.
        - Raise an error if no meta file is found.
        - Load the metadata from the meta file.
        - Filter out the meta file from the list of label files.
        - Return the metadata and filtered list of label files.""""
    
    meta_file = next((lf for lf in label_files if "meta" in lf.stem), None)

    if meta_file is None:
        raise ValueError("No meta file provided")
    meta = json.loads(meta_file.read_text())["videos"]

    label_files = [
        lf for lf in label_files if lf.suffix == extension and lf != meta_file
    ]

    return meta, label_files


def label_file_to_image(label_file: Path) -> Path:
    """Function to convert a label file path to an image file path.
    Parameters:
        - label_file (Path): Path to the label file.
    Returns:
        - Path: Path to the corresponding image file.
    Processing Logic:
        - Get the parent directory of the label file.
        - Navigate two levels up to get to the "JPEGImages" directory.
        - Append the name of the parent directory of the label file.
        - Replace the extension of the label file with ".jpg"."""
    
    return (
        label_file.parents[2]
        / "JPEGImages"
        / label_file.parent.name
        / f"{label_file.stem}.jpg"
    )


class PolyTransformer(LabelTransformer):
    def from_custom_labels(
        self, label_files: List[Path], data_files: List[Path]
        """Returns a list of DataLabel objects from the given label and data files.
        Parameters:
            - label_files (List[Path]): A list of file paths to label files.
            - data_files (List[Path]): A list of file paths to data files.
        Returns:
            - List[DataLabel]: A list of DataLabel objects containing the label and data information.
        Processing Logic:
            - Gets metadata and label files with the extension ".png".
            - Loops through each label file.
            - Gets the classes from the metadata.
            - Converts the label file to an image file.
            - Gets the shape of the image.
            - Calculates the normalization array.
            - Loops through each unique instance in the image.
            - Skips the instance if it is equal to 0.
            - Creates an instance mask.
            - Finds the contours of the instance mask.
            - Loops through each contour.
            - Converts the contour to a normalized polygon.
            - Appends a DataLabel object to the output list.
            - Returns the output list."""
        
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
