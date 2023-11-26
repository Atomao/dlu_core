import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...dataset_base import DatasetCreator
from ...utils import find_files_recursive
from ..example_configs import YoloConfig
from .utils import generate_crops, scale_from_yolo, scale_to_yolo

logging.getLogger(__name__).setLevel(logging.DEBUG)


class DatasetCreatorYolo(DatasetCreator):
    def __init__(self, config):
        if Path(config.metadata_path).exists():
            self.input_metadata = pd.read_csv(config.metadata_path)
            for stage in self.input_metadata["stage"].unique():
                (self.output_path / "labels" / stage).mkdir(parents=True, exist_ok=True)
                (self.output_path / "images" / stage).mkdir(parents=True, exist_ok=True)
        else:
            self.metadata = None

        self.images_sources = self.handle_source(config.images_sources)
        self.annotation_sources = self.handle_source(config.annotation_sources)
        self.output_path = Path(config.output_path)
        self.config = config

    @classmethod
    def default_config(cls):
        conf = YoloConfig()
        conf.set_bbox_meta_params()
        return conf

    def create_dataset_from_config(self):
        if self.metadata is None:
            raise ValueError("metadata is not set")
        for idx, row in tqdm(self.input_metadata.iterrows()):
            stem, stage = row[["stem", "stage"]]
            matching_annotation_pathes = find_files_recursive(
                self.annotation_sources, pattern=f".*{stem}*."
            )
            matching_image_pathes = find_files_recursive(
                self.images_sources, pattern=f".*{stem}*."
            )
            image_path, annotation_path = (
                matching_image_pathes[0],
                matching_annotation_pathes[0],
            )
            self.handle_image_annotation(image_path, annotation_path, stage)

    def handle_image_annotation(self, image_path, annotation_path, stage):
        divisor = self.config.crop_resolution

        image = plt.imread(image_path)
        image_height, image_width = image.shape[:2]
        if annotation_path.exists():
            with open(annotation_path, "r") as bboxes_src:
                data = bboxes_src.readlines()
            bboxes = [list(map(float, row.split(" "))) for row in data]
        else:
            print(image_path)
            bboxes = []

        if self.config.bbox_width is not None and self.config.bbox_height is not None:
            for i, bbox in enumerate(bboxes):
                if (
                    bbox[1] > self.config.bbox_width / 2
                    and bbox[1] < 1 - self.config.bbox_width / 2
                ):
                    bboxes[i][3] = self.config.bbox_width
                if (
                    bbox[2] > self.config.bbox_height / 2
                    and bbox[2] < 1 - self.config.bbox_height / 2
                ):
                    bboxes[i][4] = self.config.bbox_height

        bboxes = scale_from_yolo(
            [(x, y, w, h) for yolo_class, x, y, w, h in bboxes],
            (image_height, image_width),
        )

        padding = (
            (0, divisor - image_height % divisor),
            (0, divisor - image_width % divisor),
            (0, 0),
        )
        image = np.pad(image, padding)
        image_height, image_width = image.shape[:2]

        bboxes = scale_to_yolo(
            [(x + 1, y + 1, w + 1, h + 1) for x, y, w, h in bboxes],
            (image_height, image_width),
        )
        bboxes = np.array([(x, y, w, h, 0) for x, y, w, h in bboxes])

        crops = generate_crops(
            divisor,
            image_height,
            image_width,
            bbox_meta_params=self.config.bbox_meta_params,
        )

        output_metdata = []
        for i in range(0, image_height, divisor):
            for j in range(0, image_width, divisor):
                cropped = crops[i // divisor][j // divisor](image=image, bboxes=bboxes)

                crop_bboxes = []
                for bbox in cropped["bboxes"]:
                    x, y, w, h, c = bbox
                    crop_bboxes.append(f"{int(c)} {x} {y} {w} {h}\n")

                output_annotation_path = (
                    self.output_path
                    / "labels"
                    / stage
                    / f"{image_path.stem}_row_{i}_col_{j}_size_{divisor}"
                ).with_suffix(".txt")
                output_image_path = (
                    self.output_path
                    / "images"
                    / stage
                    / f"{image_path.stem}_row_{i}_col_{j}_size_{divisor}"
                ).with_suffix(".JPG")
                if crop_bboxes:
                    with open(output_annotation_path, "w") as dst:
                        dst.writelines(crop_bboxes)
                    empty_crop = False
                else:
                    empty_crop = True
                output_metdata.append(
                    {
                        "crop_id": output_image_path.stem,
                        "stage": stage,
                        "empty_crop": empty_crop,
                    }
                )
                plt.imsave(output_image_path, cropped["image"])
        self.output_metadata = pd.DataFrame(output_metdata)

        @staticmethod
        def output_metadata_to_input_metadata(output_metadata):
            input_metadata = (
                output_metadata.groupby("id")
                .agg({"empty_crop": "sum", "crop_id": "count", "stage": "unique"})
                .reset_index()
            )
            input_metadata["empty_crop"] = (
                input_metadata["empty_crop"].values == input_metadata["crop_id"].values
            )
            input_metadata["stage"] = input_metadata.stage.map(lambda x: x[0])
            input_metadata = input_metadata.drop(columns=["crop_id"])
            return input_metadata

        @classmethod
        def from_dataset_folder(cls, dataset_path):
            dataset_images_path = Path(dataset_path) / "images"
            dataset_labels_path = Path(dataset_path) / "labels"
            output_metadata = []
            for image_path in dataset_images_path.rglob("*.JPG"):
                stage = image_path.parent.name
                empty_crop = (
                    False
                    if (dataset_labels_path / stage / f"{image_path.stem}.txt").exists()
                    else True
                )
                original_image_stem = image_path.name.split("_row_")[0]

                output_metdata.append(
                    {
                        "id": original_image_stem,
                        "crop_id": image_path.stem,
                        "stage": stage,
                        "empty_crop": empty_crop,
                    }
                )
            output_metdata = pd.DataFrame(output_metdata)
            input_metadata = output_metadata_to_input_metadata(output_metadata)

            ds = cls(cls.default_config())
            ds.input_metadata = input_metadata
            ds.output_metadata = output_metadata
            return ds
