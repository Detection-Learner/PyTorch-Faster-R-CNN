import torch.utils.data as data
import xml.etree.ElementTree as ET
import six
import sys
import os
from PIL import Image
# Avoid IOError: image file truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import random

from ..utils.config import cfg


class DatasetError(Exception):
    pass


def _default_transform(img, gt_box):
    """
    With config file, processing the img and gt_box

    Return:
        img: Scaled `numpy.ndarray`
        img_info: Scaled (H, W, scale)
        gt_box: Scaled (n, xmin, ymin, xmax, ymax)
    """
    width, height = img.size

    if cfg.TRAIN.USE_FLIPPED == True:
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


class PascalData(data.Dataset):
    def __init__(self, VOCdevkitRoot=None, trainval=True, transform=_default_transform):

        self.trainval = trainval
        self.voc07root = os.path.join(VOCdevkitRoot, 'VOC2007')
        self.voc12root = os.path.join(VOCdevkitRoot, 'VOC2012')
        if not os.path.exists(self.voc07root):
            raise DatasetError(
                'There is not VOC2007 at {}'.format(VOCdevkitRoot))
        if not os.path.exists(self.voc12root):
            raise DatasetError(
                'There is not VOC2012 at {}'.format(VOCdevkitRoot))

        voc07list = []
        voc12list = []
        if self.trainval:
            try:
                with open(os.path.join(self.voc07root, 'ImageSets', 'Main', 'trainval.txt'), 'r') as f:
                    voc07list = f.readlines()
            except Exception as e:
                raise DatasetError("Can't open {} !".format(os.path.join(
                    self.voc07root, 'ImageSets', 'Main', 'trainval.txt')))
            try:
                with open(os.path.join(self.voc12root, 'ImageSets', 'Main', 'trainval.txt'), 'r') as f:
                    voc12list = f.readlines()
            except Exception as e:
                raise DatasetError("Can't open {} !".format(os.path.join(
                    self.voc12root, 'ImageSets', 'Main', 'trainval.txt')))
        else:
            try:
                with open(os.path.join(self.voc07root, 'ImageSets', 'Main', 'test.txt'), 'r') as f:
                    voc07list = f.readlines()
            except Exception as e:
                raise DatasetError("Can't open {} !".format(os.path.join(
                    self.voc07root, 'ImageSets', 'Main', 'test.txt')))
        self.imglist = voc07list + voc12list
        self.slice = len(voc07list)  # which id to read voc 2007
        self.transform = transform
        self.class_dict = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
                           "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12,
                           "horse": 13, "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
                           "sofa": 18, "train": 19, "tvmonitor": 20}

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        if index < self.slice:
            img_name = self.imglist[index].strip() + '.jpg'
            annotation_name = self.imglist[index].strip() + '.xml'
            img_path = os.path.join(self.voc07root, 'JPEGImages', img_name)
            annotation_path = os.path.join(
                self.voc07root, 'Annotations', annotation_name)
        else:
            img_name = self.imglist[index].strip() + '.jpg'
            annotation_name = self.imglist[index].strip() + '.xml'
            img_path = os.path.join(self.voc12root, 'JPEGImages', img_name)
            annotation_path = os.path.join(
                self.voc12root, 'Annotations', annotation_name)

        img = Image.open(img_path).convert('RGB')
        gt_box = self._read_annotation(annotation_path)

        img, img_info, gt_box = self.transform(img, gt_box)

        return img, img_info, gt_box

    def _read_annotation(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        obs = root.findall('object')
        gt_box = []
        for ob in obs:
            class_ob = ob.findall('name')
            class_name = class_ob[0].text
            bndbox_ob = ob.findall('bndbox')
            xmin_ob = bndbox_ob[0].findall('xmin')
            xmin = float(xmin_ob[0].text)
            xmax_ob = bndbox_ob[0].findall('xmax')
            xmax = float(xmax_ob[0].text)
            ymin_ob = bndbox_ob[0].findall('ymin')
            ymin = float(ymin_ob[0].text)
            ymax_ob = bndbox_ob[0].findall('ymax')
            ymax = float(ymax_ob[0].text)
            gt_box.append(
                [xmin, ymin, xmax, ymax, self.class_dict[class_name]])
        return np.array(gt_box)
