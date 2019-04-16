'''Code from torchvision with comments'''

import math

import torch.nn as nn
import torch.nn.init as init

class VGG(nn.Module):
    
    def __init__(self, features):
        super(VGG, self

def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.Maxpool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v #update the new input layer going in next layer
    return nn. Sequential(*layers)

config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg16():
    return VGG(make_layers(config, batch_norm=False))

def vgg16_bn():
    return VGG(make_layers(config, batch_norm=True))