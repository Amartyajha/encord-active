import json
from pathlib import Path
from typing import List

from encord_active.lib.labels.label_transformer import (
    BoundingBox,
    BoundingBoxLabel,
    DataLabel,
    LabelTransformer,
)


def get_meta_and_labels(label_files: List[Path], extension: str = ".json"):
    """"Returns metadata and label files from a list of provided label files.
    Parameters:
        - label_files (List[Path]): A list of label files to be processed.
        - extension (str): The file extension of the label files. Defaults to ".json".
    Returns:
        - meta (dict): A dictionary containing metadata from the provided label files.
        - label_files (List[Path]): A list of label files with the specified extension, excluding the meta file.
    Processing Logic:
        - Finds the meta file from the provided list of label files.
        - Raises a ValueError if no meta file is found.
        - Loads the metadata from the meta file.
        - Filters the label files to only include files with the specified extension.
        - Returns the metadata and filtered label files.""""
    
    meta_file = next((lf for lf in label_files if "meta" in lf.stem), None)

    if meta_file is None:
        raise ValueError("No meta file provided")
    meta = json.loads(meta_file.read_text())["videos"]

    label_files = [
        lf for lf in label_files if lf.suffix == extension and lf != meta_file
    ]

    return meta, label_files


def label_file_to_image(label_file: Path) -> Path:
    """Returns the path to the corresponding image file given a label file.
    Parameters:
        - label_file (Path): Path to the label file.
    Returns:
        - Path: Path to the corresponding image file.
    Processing Logic:
        - Get the parent directory of the parent directory of the label file.
        - Append "JPEGImages" to the path.
        - Append the name of the parent directory of the label file to the path.
        - Append the name of the label file without the extension and with ".jpg" appended to the path."""
    
    return (
        label_file.parents[2]
        / "JPEGImages"
        / label_file.parent.name
        / f"{label_file.stem}.jpg"
    )


class BBoxTransformer(LabelTransformer):
    def from_custom_labels(
        self, label_files: List[Path], data_files: List[Path]
        """"""
        
    ) -> List[DataLabel]:
        meta, label_files = get_meta_and_labels(label_files, extension=".json")

        out = []
        for label_file in label_files:
            classes = meta[label_file.parent.name]["objects"]

            labels = json.loads(label_file.read_text())
            image_file = label_file_to_image(label_file)

            for instance_id, bbox in labels.items():
                out.append(
                    DataLabel(
                        abs_data_path=image_file,
                        label=BoundingBoxLabel(
                            class_=classes.get(instance_id, {}).get(
                                "category", "unknown"
                            ),
                            bounding_box=BoundingBox(**bbox),
                        ),
                    )
                )
        return out
