from ..model_zoo.vgg import *

def Network(net_name, feature_layer=None, is_det=False):

    if is_det == True:
        return _get_detection_network(net_name, feature_layer)
    else:
        return _get_network(net_name)

def _get_detection_network(net_name, feature_layers=None):

    basic_network, rcnn = eval(net_name + '(feature_layers={})'.format(feature_layers))

    return basic_network, rcnn
