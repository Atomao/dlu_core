import albumentations as A


class Config:
    def __init__(self, conf):
        self.metadata_path = "Not set"
        self.output_path = "Not set"
        self.images_sources = "Not set"
        self.annotation_sources = "Not set"
        self.annotations_extension = "Not set"
        self.crop_resolution = "Not set"
        

    def from_dict(self, conf):
        for key, value in conf.items():
            setattr(self, key, value)

    def __str__(self):
        result_string = "Config:\n"
        for key, value in self.__dict__.items():
            result_string += f"\t{key}={value}\n"
        return result_string

    def __repr__(self):
        return self.__str__()


class YoloConfig(Config):
    def __init__(self):
        self.metadata_path = "/home/danylo/Desktop/agroscout/sugarcane_billbug/notebooks/billbug_dataset1.csv"
        self.output_path = "/home/danylo/Desktop/agroscout/sugarcane_billbug/notebooks/datasets/dataset2"
        self.images_sources = (
            "/home/danylo/Desktop/agroscout/sugarcane_billbug/data/raw/images/"
        )
        self.annotation_sources = "/home/danylo/Desktop/agroscout/sugarcane_billbug/data/raw/labeled_iterartion1/annotations billbug/"
        self.annotations_extension = ".txt"
        self.crop_resolution = 640


    def set_bbox_meta_params(self, min_visibility=None, bbox_width=None, bbox_height=None):
        min_visibility = 0.3 if min_visibility is None else min_visibility
        self.bbox_meta_params = A.BboxParams(format="yolo", min_visibility=min_visibility)
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height
