# --------------------------------------------
# Faster R-CNN / VGG module
# Written by Jiyun Cui
# Modified by Yuanshun Cui
# --------------------------------------------

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGGBasicNetwork(nn.Module):
    # the convolutions of VGG network
    def __init__(self, feature_modules):
        super(VGGBasicNetwork, self).__init__()
        [self.feat_modules, self.feat_strides, self.out_dim] = feature_modules
        self.feat_modules = nn.ModuleList(self.feat_modules)
        self._initialize_weights()

    def forward(self, x):
        features = []
        for i, ele_module in enumerate(self.feat_modules):
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
        self.classifier = nn.Sequential(
            nn.Linear(in_channel * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self._initialize_weights()
        self.out_dim = 4096

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
    feat_strides = []
    # corresponding channels
    out_dim = []
    stride = 1

    # the last feature of vgg network must in the output feature list
    if cfg[-1][1] not in feature_layers:
        feature_layers.append(cfg[-2][1])

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
            feat_strides.append(stride)
            out_dim.append(in_channels)
            layers = []
    return feature_modules, feat_strides, out_dim


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'A_name': ['conv1_1', 'pool1', 'conv2_1', 'pool2', 'conv3_1', 'conv3_2',
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


def load_weights(models=None, features=None):
    from collections import OrderedDict
    features_dict = OrderedDict()
    classifier_dict = OrderedDict()

    f_keys = features.keys()

    for i, (k, v) in enumerate(models.items()):
        if 'features' in k:
            features_dict[f_keys[i]] = v
        elif '6' not in k:
            name = k
            classifier_dict[name] = v
    return features_dict, classifier_dict


def vgg11(pretrained=False, feature_layers=None, **kwargs):
    """ VGG 11-layer model (configuration 'A')
    Args:
    :param pretrained: todo
    :param feature_layers: return specific features named by feature_layers
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['A'], cfg['A_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg11'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['A'], cfg['A_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg11_bn'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['B'], cfg['B_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg13'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['B'], cfg['B_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg13_bn'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['D'], cfg['D_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg16'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['D'], cfg['D_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg16_bn'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['E'], cfg['E_name']), feature_layers))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg19'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

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
    features = VGGBasicNetwork(make_feat_modules(
        zip(cfg['E'], cfg['E_name']), feature_layers, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGGClassifier(**kwargs)

    if pretrained:
        models = model_zoo.load_url(model_urls['vgg19_bn'])
        features_dict, classifier_dict = load_weights(
            models=models, features=features.state_dict())
        features.load_state_dict(features_dict)
        state_dict = classifier.state_dict()
        state_dict.update(classifier_dict)
        classifier.load_state_dict(state_dict)

    return features, classifier
