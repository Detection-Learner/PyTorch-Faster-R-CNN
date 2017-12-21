import numpy as np
import random
import torch
import torch.utils.data as data
import math
from PIL import Image
from ...utils.config import cfg


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
            box = np.zeros((1, 5), dtype=np.float32)
            for i, x in enumerate(info):
                box[0][i] = int(x)
            assert box[0][0] < box[0][2] and box[0][1] < box[0][3], '{} box size error'.format(
                ann_path)
            all_boxes = np.vstack((all_boxes, box))
            line = fp.readline()

    return all_boxes


def detection_collate(batch):
    """
    Return:

    images: batch_size * 3 * max_height * max_width, tensor

    img_infos: batch_size * 3, np.ndarray

    gt_boxes: batch_size * gt_i, list

    """
    imgs, img_infos, gt_boxes = zip(*batch)
    batch_size = len(imgs)

    # max [height, width]
    max_shape = np.array([[im_info[0], im_info[1]]
                          for im_info in img_infos]).max(axis=0)
    images = np.empty((batch_size, int(max_shape[0]), int(
        max_shape[1]), 3), dtype=np.float32)
    images.fill(128)
    for i in range(batch_size):
        img = imgs[i]
        images[i, 0: int(img_infos[i][0]), 0: int(img_infos[i][1]), :] = img
    images = np.ascontiguousarray(images)

    # normalize
    #images = (images - cfg.PIXEL_MEANS)  # / 128.0
    images = (images - 128.) / 128.
    images = images.transpose((0, 3, 1, 2))
    images = np.ascontiguousarray(images).astype(np.float32)

    images = torch.from_numpy(images)
    img_infos = np.vstack(img_infos)

    return images, img_infos, gt_boxes


def transform(img, gt_box):

    width, height = img.size

    if cfg.TRAIN.USE_FLIPPED ==True:
        seed = random.uniform(0, 1)
        if seed < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # flip boxes
            oldx1 = gt_box[:, 0].copy()
            oldx2 = gt_box[:, 2].copy()
            gt_box[:, 0] = width - oldx2 - 1
            gt_box[:, 2] = width - oldx1 - 1

    img_size_min = np.min(img.size)
    img_size_max = np.max(img.size)

    scale_id = random.randint(0, len(cfg.TRAIN.SCALES) - 1)
    target_size = cfg.TRAIN.SCALES[scale_id]
    img_scale = float(target_size) / float(img_size_min)
    if np.round(img_scale * img_size_max) > cfg.TRAIN.MAX_SIZE:
        img_scale = float(cfg.TRAIN.MAX_SIZE) / float(img_size_max)

    img = img.resize((int(round(width * img_scale)),
                      int(round(height * img_scale))), Image.BILINEAR)
    im_info = np.array(
        [round(height * img_scale), round(width * img_scale), img_scale])
    gt_box[:, 0:4] *= img_scale

    return np.array(img, dtype=np.float32), im_info, gt_box


class ImageLoader(data.Dataset):

    def __init__(self, img_path, ann_path):

        self.img_list = _read_data_list(img_path)
        self.ann_list = _read_data_list(ann_path)

    def __getitem__(self, index):

        img_path = self.img_list[index]
        ann_path = self.ann_list[index]

        img = Image.open(img_path).convert('RGB')
        gt_box = _load_annitation(ann_path)
        img, im_info, gt_box = transform(img, gt_box)

        return img, im_info, gt_box

    def __len__(self):

        return len(self.img_list)
