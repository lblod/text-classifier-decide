import json
from datasets import Dataset
import numpy as np
from abc import ABC, abstractmethod


class LabeledData(ABC):
    """Base class for labeled data handling.

    Args:
        file_path (str): Path to the JSON file containing labeled data.

    Attributes:
        data (list[dict]): List of tasks read from the JSON file.
        label2id (dict): Mapping from label names to their corresponding IDs.
        id2label (dict): Mapping from label IDs to their corresponding names.

    """
    def __init__(self, file_path: str):
        super().__init__()
        self.data = self.read_data(file_path)
        self.label2id, self.id2label = self.generate_label_map()

    def read_data(self, file_path: str) -> list[dict]:
        ## Todo this should straight from LD
        with open(file_path) as fd:
            data = json.loads(fd.read())
            print(len(data))
        return data

    def generate_label_map(self):
        label_set = {l for task in self.data if task["annotations"][0]["result"] for l in task["annotations"][0]["result"][0]["value"]["choices"]}
        id2label = {idx: label for idx, label in enumerate(label_set)}
        label2id = {label: idx for idx, label in enumerate(label_set)}
        return label2id, id2label

    @abstractmethod
    def format(self) -> Dataset:
        """Format the data into a Hugging Face Dataset."""
        pass


class SingleLabelData(LabeledData):
    def format(self) -> Dataset:
        """ Format the data into a Hugging Face Dataset for single-label classification. """
        return Dataset.from_list([
            {
                "text": task["data"]["text"],
                "label": self.label2id[task["annotations"][0]["result"][0]["value"]["choices"][0]]
            }
            for task in self.data if task["annotations"][0]["result"]
        ]).class_encode_column("label").train_test_split(test_size=0.1, stratify_by_column="label")


class MultiLabelData(LabeledData):
    def format(self) -> Dataset:
        """ Format the data into a Hugging Face Dataset for multi-label classification. """
        return Dataset.from_list([
            {
                "text": task["data"]["text"],
                "label": np.isin(
                    np.arange(len(self.label2id)),
                    [self.label2id[l] for l in task["annotations"][0]["result"][0]["value"]["choices"]]
                ).astype(float).tolist()
            }
            for task in self.data if task["annotations"][0]["result"]
        ]).dataset.train_test_split(test_size=0.1)


def get_dataset_cls(problem_type: str):
    if problem_type == 'single_label_classification':
        return SingleLabelData
    elif problem_type == 'multi_label_classification':
        return MultiLabelData
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")