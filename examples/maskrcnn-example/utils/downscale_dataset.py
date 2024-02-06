import copy
import json
import shutil
from pathlib import Path

import cv2
from tqdm.auto import tqdm

target_width = 1024
target_height = 1024
data_folder_path = "/path/to/ea-project"
annotations_path = "/path/to/annotation/file"


def resize_images(root_folder: Path, target_width: int, target_height: int):
    """Parameters:
        - root_folder (Path): The root folder containing the images to be resized.
        - target_width (int): The desired width of the resized images.
        - target_height (int): The desired height of the resized images.
    Returns:
        - None: The function does not return any value, it simply resizes the images and saves them in a new folder.
    Processing Logic:
        - Deletes any existing target data folder.
        - Creates a new target data folder with the specified width and height in the name.
        - Iterates through the images in the data folder.
        - Creates a new folder for each image unit in the target data folder.
        - Resizes each image to the specified width and height using OpenCV.
        - Saves the resized image in the corresponding folder in the target data folder."""
    
    data_path = Path(root_folder / "data")
    target_data_path = Path(root_folder / f"data_{target_width}x{target_height}")

    if target_data_path.is_dir():
        shutil.rmtree(target_data_path)
    target_data_path.mkdir()

    for lr_item in tqdm(
        data_path.iterdir(),
        desc="Resizing images",
        total=len(list(data_path.glob("*"))),
    ):
        if lr_item.is_dir():
            data_units = lr_item / "images"
            target_data_units = (
                target_data_path / lr_item.as_posix().split("/")[-1] / "images"
            )
            target_data_units.mkdir(parents=True)
            for item in data_units.glob("*.*"):
                if item.as_posix().endswith(
                    (".png", ".jpg", ".jpeg", "tiff", ".tif", ".bmp", ".gif")
                ):
                    target_file_path = target_data_path / "/".join(
                        item.as_posix().split("/")[-3:]
                    )
                    try:
                        img = cv2.imread(item.as_posix())
                        img_resized = cv2.resize(
                            img,
                            (target_width, target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    except Exception:
                        print(f"Corrupted file: {item.as_posix}")
                        continue
                    cv2.imwrite(target_file_path.as_posix(), img_resized)
                else:
                    print(f"Not supported file: {item.as_posix()}")


def resize_coco_annotations(
    coco_annotation_file_path: Path,
    images_folder_name: str,
    target_width: int,
    target_height: int,
):
    """Resizes COCO annotations to a specified target width and height.
    Parameters:
        - coco_annotation_file_path (Path): Path to the COCO annotation file.
        - images_folder_name (str): Name of the folder containing the images.
        - target_width (int): Target width for resizing.
        - target_height (int): Target height for resizing.
    Returns:
        - target_coco_annotation_file_path (str): Path to the resized COCO annotation file.
    Processing Logic:
        - Creates a new file path for the resized COCO annotation file.
        - Loads the COCO annotation file.
        - Creates a deep copy of the COCO annotation.
        - Resizes the image info and annotations.
        - Writes the resized COCO annotation to the new file path."""
    
    target_coco_annotation_file_path = (
        coco_annotation_file_path.with_suffix("").as_posix()
        + f" {target_width}x{target_height}"
        + coco_annotation_file_path.suffix
    )

    coco_annotation = json.loads(coco_annotation_file_path.read_text())
    target_coco_annotation = copy.deepcopy(coco_annotation)

    for image in tqdm(
        iterable=target_coco_annotation["images"],
        desc="Resizing image info",
        total=len(target_coco_annotation["images"]),
    ):
        image["file_name"] = "/".join(
            (images_folder_name, image["file_name"].split("/", 1)[1])
        )
        image["width"] = target_width
        image["height"] = target_height

    for annotation in tqdm(
        target_coco_annotation["annotations"], desc="Resizing annotations"
    ):
        width_scaling = (
            target_width / coco_annotation["images"][annotation["image_id"]]["width"]
        )
        height_scaling = (
            target_height / coco_annotation["images"][annotation["image_id"]]["height"]
        )

        annotation["area"] = target_width * target_height
        annotation["bbox"] = [
            annotation["bbox"][0] * width_scaling,
            annotation["bbox"][1] * height_scaling,
            annotation["bbox"][2] * width_scaling,
            annotation["bbox"][3] * height_scaling,
        ]
        for segmentation_item in annotation["segmentation"]:
            for i, point in enumerate(segmentation_item):
                if i % 2 == 0:
                    segmentation_item[i] = point * width_scaling
                else:
                    segmentation_item[i] = point * height_scaling

    with open(target_coco_annotation_file_path, "w", encoding="utf-8") as fp:
        json.dump(target_coco_annotation, fp)


resize_images(
    Path(data_folder_path),
    target_width,
    target_height,
)

resize_coco_annotations(
    Path(annotations_path),
    f"data_{target_width}x{target_height}",
    target_width,
    target_height,
)
