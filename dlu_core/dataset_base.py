import abc
import logging
from pathlib import Path

logging.getLogger(__name__).setLevel(logging.DEBUG)


class DatasetCreator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def create_dataset_from_config(self):
        raise NotImplementedError

    @abc.abstractmethod
    def handle_image_annotation(self, image_path, annotation_path, stage):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dataset_folder(cls, dataset_path):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def default_config(cls):
        raise NotImplementedError
