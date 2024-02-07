import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlu_core.object_detection.yolo.utils import load_yolo_annotations_file

from ..base.visualizer import BaseVisualizer


class ObjectDetectionVisualizer(BaseVisualizer):
    def _init_visualizer(self):
        self.name = self.config.get("name", "VisualizerObjectDetection")
        self.default_value_name = self.config.get("default_value_name", "default")
        self.default_value_color = self.config.get("default_value_color", (0, 0, 0))
        self.default_value_idx = self.config.get("default_value_idx", 0)

        self.classes_colors = self.config.get(
            "classes_colors", {self.default_value_name: self.default_value_color}
        )
        self.idxs_classes = self.config.get(
            "idxs_classes", {self.default_value_idx: self.default_value_name}
        )
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.config.get("font_scale", 1)
        self.font_thickness = self.config.get("font_thickness", 2)
        self.line_thickness = self.config.get("line_thickness", 2)
        self.line_type = self.config.get("line_type", cv2.LINE_AA)
        self.default_name = "default"

    def draw_bounding_boxes(self, image, boxes, labels, scores, show_label):
        image = image.copy()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            class_name = self.idxs_classes.get(label, self.default_value_name)
            color = self.classes_colors.get(class_name, self.default_value_color)
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(
                image, (x1, y1), (x2, y2), color, self.line_thickness, self.line_type
            )
            if show_label:
                text = f"{self.idxs_classes[label]}: {score:.2f}"
                cv2.putText(
                    image,
                    text,
                    (x1, y1 - 5),
                    self.font,
                    self.font_scale,
                    color,
                    self.font_thickness,
                    self.line_type,
                )
        return image


class YoloVisualizer(ObjectDetectionVisualizer):
    def _load_yolo_annotations(self, annotations_path, height, width):
        return load_yolo_annotations_file(annotations_path, height, width)

    def visualize_yolo_output(self, image, detection_df, show_label=False):
        boxes = detection_df[["xmin", "ymin", "xmax", "ymax"]].values
        labels = detection_df["label"].values
        scores = detection_df["score"].values
        image = self.draw_bounding_boxes(
            image, boxes, labels, scores, show_label=show_label
        )
        return image

    def visualize_yolo_annotation(self, image, annotations_path, show_label=False):
        height, width = image.shape[:2]
        annotations = self._load_yolo_annotations(annotations_path, height, width)
        boxes = annotations[["xmin", "ymin", "xmax", "ymax"]].values
        labels = annotations["label"].values
        image = self.draw_bounding_boxes(
            image, boxes, labels, np.ones(len(labels)), show_label=show_label
        )
        return image

    def process_yolo_dir(
        self,
        images_dir,
        annotations_dir,
        output_dir,
        model=None,
        N=10,
        show_label=False,
    ):
        images_dir, annotations_dir, output_dir = map(
            Path, images_dir, annotations_dir, output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(self.classes_colors.keys()) < 2:
            old_classes_colors = self.idxs_classes.copy()
            for image_path in images_dir.glob("*.JPG"):
                if image_path.is_file():
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.classes_colors = {
                        self.default_value_name: (0, 0, 255)
                    }  # Blue is annotation color
                    image = self.visualize_yolo_annotation(
                        image,
                        annotations_dir / image_path.with_suffix("txt").name,
                        show_label,
                    )
                    self.classes_colors = {
                        self.default_value_name: (255, 0, 0)
                    }  # Red is detection color
                    image = self.visualize_yolo_output(
                        image, model.predict(str(image_path)), show_label
                    )
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_dir / image_path.name), image)
            self.classes_colors = old_classes_colors
        else:
            raise NotImplementedError(
                "process_dir not implemented for multiclass detection"
            )
