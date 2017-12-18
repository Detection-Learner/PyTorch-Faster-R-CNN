#-----------------------------
# PyTorch Faster R-CNN
# Written by Hongyu Pan
#-----------------------------

import numpy as np

class ROIDataLayer(object):
    """
    Faster R-CNN data layer used for training.
    """

    def __init__(self, roidb, num_classes, random=False):
