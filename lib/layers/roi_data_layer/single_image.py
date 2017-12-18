import numpy as as np
import torch
import torch.utils.data as data
import math
from PIL import Image

def _read_data_list(file_path):

    lists = []
    with open(file_path, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists

def _load_annitation(ann_path):
    """
    load the annotations.
    for each box, the format is [x1, y1, x2, y2, label].
    and x1 and x2 belong to [0, w), y1 and y2 belong to [0, h).
    """
    all_boxes = np.zeros((0, 5), dtype=np.int32)
    with open(ann_path, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            box = np.zeros((1, 5), dtype=np.int32)
            for i, x in enumerate(info):
                box[i] = int(x)
            assert box[0] < box[2] and box[1] < box[3]
            all_boxes = np.vstack((all_boxes, box))
            line = fp.readline()

    return all_boxes

class SingleImage(data.Dataset):

    def __init__(self, img_path, ann_path, transformer=None):

        self.img_list = _read_data_list(img_path)
        self.ann_list = _read_data_list(ann_path)

    def __getitem__(self, indx):

        img_path = self.img_list[index]
        ann_path = self.ann_list[index]

        img = Image.open(img_path).convert('RGB')
        gt_boxes = _load_annitation(ann_path)

        if seld.transform is not None:
            img = self.transform(img)

        im_info = []

        return img, im_info, gt_boxes
