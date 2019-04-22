'''Code from torchvision with comments'''

import math

import torch.nn as nn
import torch.nn.init as init

class VGG(nn.Module): #make subclass VGG - the super class is inside the parentheses. 
    
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        #final classifier: dropout, fc, relu, dropout, fc, relu, fc
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        
        for m in self.modules(): #iterates through the modules in the network
            if isinstance(m, nn.Conv2d): #looks for the conv kernels in all the modules. isinstance is used for subclasses, type() does not acount for such
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n)) #initializes the veights as a normal dist
                m.bias.data.zero_() #initializes the bias = zero
    
    def forward(self, x):
        '''Performs the forward pass of the network'''
        x = self.features(x) #forward pass of layers
        x = x.view(x.size(0), -1) #x.view reshapes pytorch tensor is have the first dim shape, and all other dims combined for the second dim
        x = self.classifier(x) #executes final classifier layer
        return x
           

def make_layers(config, batch_norm=False):
    '''Inputs: an array of shape numbers and Maxpools, outputs: architectures'''
    layers = []
    in_channels = 3 #RGB
    for v in config:
        if v == 'M': #MaxPool layer
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else: #is v is given as out_channels
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: #add layer of conv, batchnorm, relu
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else: #add layer of conv, relu
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v #update the new input layer going in next layer
    return nn.Sequential(*layers)

config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg16():
    return VGG(make_layers(config, batch_norm=False))

def vgg16_bn():
    return VGG(make_layers(config, batch_norm=True))