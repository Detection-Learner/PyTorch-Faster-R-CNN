# --------------------------------------------
# Faster R-CNN / VGG module
# Written by Jiyun Cui
# --------------------------------------------

import torch.nn as nn
import math

__all__ = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGGBasicNetwork(nn.Module):
    # the convolutions of VGG network
    def __init__(self, feature_modules):
        super(VGGBasicNetwork, self).__init__()
        [self.feat_modules, self.feat_stride, self.out_dim] = feature_modules
        self.feat_modules = nn.ModuleList(self.feat_modules)
        self._initialize_weights()

    def forward(self, x):
        features = []
        for ele_module in self.feat_modules:
            x = ele_module(x)
            features.append(x)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGGClassifier(nn.Module):
    # the fully connection network
    def __init__(self, in_channel=512):
        super(VGGClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channel * 7 * 7, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_feat_modules(cfg, feature_layers, batch_norm=False):
    # append layers for every part
    layers = []
    in_channels = 3
    # sequential layers
    feature_modules = []
    # downsample multipler
    feat_stride = []
    # corresponding channels
    out_dim = []
    stride = 1

    # the last feature of vgg network must in the output feature list
    if cfg[-1][1] not in feature_layers:
        feature_layers.append(cfg[-1][1])

    for (v, layer_name) in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            stride = stride * 2
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        if layer_name in feature_layers:
            feature_modules.append(nn.Sequential(*layers))
            feat_stride.append(stride)
            out_dim.append(in_channels)
            layers = []
    return feature_modules, feat_stride, out_dim

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'A_name': ['conv1_1', 'pool1', 'conv2_1', 'pool2', 'conv3_1','conv3_2',
               'pool3', 'conv4_1', 'conv4_2', 'pool4', 'conv5_1', 'conv5_2'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
               'conv3_2', 'pool3', 'conv4_1', 'conv4_2', 'pool4', 'conv5_1', 'conv5_2'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
               'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'E_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
               'conv3_4', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4', 'conv5_1', 'conv5_2',
               'conv5_3', 'conv5_4'],
}


def vgg11(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 11-layer model (configuration 'A')
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['A'], cfg['A_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg11_bn(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 11-layer model (configuration 'A') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['A'], cfg['A_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg13(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 13-layer model (configuration 'B')
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['B'], cfg['B_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg13_bn(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 13-layer model (configuration 'B') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['B'], cfg['B_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg16(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 16-layer model (configuration 'D')
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['D'], cfg['D_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg16_bn(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 16-layer model (configuration 'D') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['D'], cfg['D_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg19(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 19-layer model (configuration 'E')
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['E'], cfg['E_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


def vgg19_bn(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 19-layer model (configuration 'E') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(zip(cfg['E'], cfg['E_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)
    return features, classifier


