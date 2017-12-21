import numpy as np

def bbox_overlaps(boxes, query_boxes):

    box_areas = ((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)).reshape(-1, 1)
    query_areas = ((query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)).reshape(1, -1)

    iw = (np.minimum(boxes[:, 2:3], query_boxes[:, 2:3].transpose()) - np.maximum(boxes[:, 0:1], query_boxes[:, 0:1].transpose()) + 1).clip(min=0)
    ih = (np.minimum(boxes[:, 3:4], query_boxes[:, 3:4].transpose()) - np.maximum(boxes[:, 1:2], query_boxes[:, 1:2].transpose()) + 1).clip(min=0)

    ua = box_areas + query_areas - iw * ih
    iou = iw * ih / ua

    return iou

def bbox_intersections(boxes, query_boxes):

    query_areas = ((query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)).reshape(1, -1)

    iw = (np.minimum(boxes[:, 2:3], query_boxes[:, 2:3].transpose()) - np.maximum(boxes[:, 0:1], query_boxes[:, 0:1].transpose()) + 1).clip(min=0)
    ih = (np.minimum(boxes[:, 3:4], query_boxes[:, 3:4].transpose()) - np.maximum(boxes[:, 1:2], query_boxes[:, 1:2].transpose()) + 1).clip(min=0)

    intersection = iw * ih / query_areas

    return intersection
