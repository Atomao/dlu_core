def center_xywh2xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def xywh2xyxy(x, y, w, h):
    return x, y, x + w, y + h


def xyxy2xywh(x1, y1, x2, y2):
    return x1, y1, x2 - x1, y2 - y1
