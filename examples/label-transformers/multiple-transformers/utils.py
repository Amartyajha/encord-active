import json
from pathlib import Path
from typing import List


def get_meta_and_labels(label_files: List[Path], extension: str = ".json"):
    """Get meta data and labels from provided label files.
    Parameters:
        - label_files (List[Path]): List of paths to label files.
        - extension (str): Optional. File extension for label files. Defaults to ".json".
    Returns:
        - meta (dict): Dictionary containing meta data for videos.
        - label_files (List[Path]): List of paths to label files.
    Processing Logic:
        - Find and load meta file.
        - Raise error if no meta file is provided.
        - Filter label files to only include files with specified extension.
        - Return meta data and filtered label files.
    Example:
        meta, label_files = get_meta_and_labels(["video1.json", "video2.json", "meta.json"])
        # meta = {"videos": {...}}
        # label_files = ["video1.json", "video2.json"]"""
    
    meta_file = next((lf for lf in label_files if "meta" in lf.stem), None)

    if meta_file is None:
        raise ValueError("No meta file provided")
    meta = json.loads(meta_file.read_text())["videos"]

    label_files = [
        lf for lf in label_files if lf.suffix == extension and lf != meta_file
    ]

    return meta, label_files


def label_file_to_image(label_file: Path) -> Path:
    """Function:
        This function converts a label file to an image file.
    Parameters:
        - label_file (Path): The path to the label file.
    Returns:
        - Path: The path to the converted image file.
    Processing Logic:
        - Get parent directory of label file.
        - Get grandparent directory of label file.
        - Get JPEGImages directory from grandparent directory.
        - Get parent directory name from label file.
        - Get stem of label file.
        - Combine all components to create path to image file.
    Example:
        label_file_to_image(Path("data/labels/label_1.txt"))
        # Returns Path("data/JPEGImages/labels/label_1.jpg")"""
    
    return (
        label_file.parents[2]
        / "JPEGImages"
        / label_file.parent.name
        / f"{label_file.stem}.jpg"
    )
