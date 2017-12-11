import troch.nn as nn
import math

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG_Convs(nn.Module):
    def __init__(self, features):
        super(VGG_Convs, self).__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


class VGG_Classifier(nn.Module):
    def __init__(self, in_channel=512, num_classes=1000):
        self.fc1 = nn.Linear(in_channel * 7 * 7, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


def make_layers(cfg, feature_layer, batch_norm=False):
    layers = []
    in_channels = 3
    for (v, layer_name) in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        if layer_name == feature_layer:
            break
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A_name': ['conv1_1', 'pool1', 'conv2_1', 'pool2', 'conv3_1','conv3_2',
               'pool3', 'conv4_1', 'conv4_2', 'pool4', 'conv5_1', 'conv5_2', 'pool5'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
               'conv3_2', 'pool3', 'conv4_1', 'conv4_2', 'pool4', 'conv5_1', 'conv5_2', 'pool5'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
               'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'pool5'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'E_name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
               'conv3_4', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4', 'conv5_1', 'conv5_2',
               'conv5_3', 'conv5_4', 'pool5'],
}


def vgg11(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 11-layer model (configuration 'A')
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['A'], cfg['A_name']), feature_layer))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg11_bn(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 11-layer model (configuration 'A') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['A'], cfg['A_name']), feature_layer, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg13(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 13-layer model (configuration 'B')
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['B'], cfg['B_name']), feature_layer))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg13_bn(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 13-layer model (configuration 'B') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['B'], cfg['B_name']), feature_layer, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg16(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 16-layer model (configuration 'D')
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['D'], cfg['D_name']), feature_layer))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg16_bn(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 16-layer model (configuration 'D') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['D'], cfg['D_name']), feature_layer, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg19(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 19-layer model (configuration 'E')
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['E'], cfg['E_name']), feature_layer))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


def vgg19_bn(pretrained=False, feature_layer='pool5', **kwargs):
    """ VGG 19-layer model (configuration 'E') with batch normalization
    Args:
    :param pretrained: todo
    :param feature_layer: construct the vgg network till feature_layer
    :param kwargs: in_channels, num_classes
    :return: both model of convolutions and classifier
    """
    # the model of convolutional layers for features extraction
    features = VGG_Convs(make_layers(zip(cfg['E'], cfg['E_name']), feature_layer, batch_norm=True))
    # the model of fully convolutional layers for classification
    classifier = VGG_Classifier(**kwargs)
    return features, classifier


