import albumentations as A
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
import random

def plot_random_yolo_crop(images_folder_path, annotations_folder_path):
    images_path = Path(images_folder_path) 
    annotations_path = Path(annotations_folder_path)

    a_path = random.sample(list(annotations_path.rglob("*.txt")),k=1)[0]
    i_path = images_path / a_path.parent.name / a_path.with_suffix(".JPG").name
    if i_path.exists() and a_path.exists():
        visualize_yolo(i_path, a_path)
    else:
        print(f"Not exist image, annot: ", i_path.exists(), a_path.exists())


def visualize_yolo(image_path, label_path):
    img = plt.imread(image_path, label_path)
    dh, dw, _ = img.shape
    
    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()
    
    for dt in data:
    
        # Split string to float
        c, x, y, w, h = map(float, dt.split(' '))
    
        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        if c == 0:
            color = (0, 0, 255)
        elif c == 1:
            color = (255, 0, 0)
        cv2.rectangle(img, (l, t), (r, b), color, 20)
    
    plt.imshow(img)
    plt.show()


def generate_crops(yolo_resolution, image_height, image_width, bbox_meta_params):
    row_num = image_height // yolo_resolution
    col_num = image_width // yolo_resolution
    crops = [[None for _ in range(col_num)] for _ in range(row_num)]

    for i in range(row_num):
        y_min = i * yolo_resolution
        y_max = (i + 1) * yolo_resolution
        if i == row_num - 1:
            y_min = -1 * yolo_resolution + image_height
            y_max = image_height
        for j in range(col_num):
            x_min = j * yolo_resolution
            x_max = (j + 1) * yolo_resolution
            if j == col_num - 1:
                x_min = -1 * yolo_resolution + image_width
                x_max = image_width
            crops[i][j] = A.Compose(
                [
                    A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
                ],
                bbox_params=bbox_meta_params,
            )
    return crops


def scale_from_yolo(bboxes, size):
    h, w = size
    for idx, bbox in enumerate(bboxes):
        bboxes[idx] = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    return bboxes


def scale_to_yolo(bboxes, size):
    h, w = size
    for idx, bbox in enumerate(bboxes):
        bboxes[idx] = bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h
    return bboxes
