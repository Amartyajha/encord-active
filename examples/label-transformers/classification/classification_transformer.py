from pathlib import Path
from typing import List

from encord_active.lib.labels.label_transformer import (
    ClassificationLabel,
    DataLabel,
    LabelTransformer,
)


class ClassificationTransformer(LabelTransformer):
    def from_custom_labels(self, _, data_files: List[Path]) -> List[DataLabel]:
        """Creates a list of DataLabels from a list of data files.
        Parameters:
            - self (type): The class instance.
            - _ (type): Unused parameter.
            - data_files (List[Path]): A list of Path objects representing data files.
        Returns:
            - List[DataLabel]: A list of DataLabel objects.
        Processing Logic:
            - Create a DataLabel object for each file.
            - Use the file's parent directory name as the classification label.
            - Add the DataLabel object to the list.
            - Return the list of DataLabels."""
        
        return [
            DataLabel(f, ClassificationLabel(class_=f.parent.name)) for f in data_files
        ]
