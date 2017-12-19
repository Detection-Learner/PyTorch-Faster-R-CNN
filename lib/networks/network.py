from ..model_zoo.vgg import *

def Network(net_name, feature_layers=None, is_det=False, use_fpn=False):

    if is_det == True:
        if use_fpn:
            in_channels = 256
        else:
            in_channels = 512
        return _get_detection_network(net_name, feature_layers, in_channels)
    else:
        return _get_network(net_name)

def _get_detection_network(net_name, feature_layers=None, in_channels=512):

    basic_network, rcnn = eval(net_name + '(feature_layers={}, in_channel={})'.format(feature_layers, in_channels))

    return basic_network, rcnn
