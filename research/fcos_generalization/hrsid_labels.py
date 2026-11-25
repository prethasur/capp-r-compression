"""HRSID label conversion used by the FCOS data loader."""

from __future__ import annotations


def yolo_line_to_coco_bbox(line: str, width: int, height: int) -> tuple[int, list[float]]:
    """Convert a YOLO box or segmentation polygon to an axis-aligned COCO box."""
    values = [float(token) for token in line.split()]
    if len(values) < 5:
        raise RuntimeError("official-validation label has fewer than five values")
    class_id = int(values[0])
    coordinates = values[1:]
    if len(coordinates) == 4:
        x_center, y_center, box_width, box_height = coordinates
        w = box_width * width
        h = box_height * height
        x = x_center * width - w / 2
        y = y_center * height - h / 2
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x_min, x_max = min(xs) * width, max(xs) * width
        y_min, y_max = min(ys) * height, max(ys) * height
        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
    else:
        raise RuntimeError("official-validation label is neither a YOLO box nor polygon")
    if w < 0 or h < 0:
        raise RuntimeError("official-validation label produced a negative-size box")
    return class_id, [x, y, w, h]
