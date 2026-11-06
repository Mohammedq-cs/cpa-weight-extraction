"""
Convolutional neural network with five convolutional layer and two fully-
connected layers afterwards

@author: dmholtz
"""

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1. convolutional layer
        # sees 32x32x3 image tensor, i.e 32x32 RGB pixel image
        # outputs 32 filtered images, kernel-size is 3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        # 2. convolutional layer
        # sees 16x16x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 32 filtered images, kernel-size is 3
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        # 3. convolutional layer
        # sees 8x8x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 64 filtered images, kernel-size is 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128 ,3, padding = 1)
        self.conv6_bn = nn.BatchNorm2d(128)
        
        # Definition of the MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1. fully-connected layer
        # Input is a flattened 4*4*64 dimensional vector
        # Output is 500 dimensional vector
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        # defintion of dropout (dropout probability 25%)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)

    def forward(self, x):
        # Pass data through a sequence of 3 convolutional layers
        # Firstly, filters are applied -> increases the depth
        # Secondly, Relu activation function is applied
        # Finally, MaxPooling layer decreases width and height
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.dropout20(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
        x = self.dropout30(x)
        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.pool(self.conv6_bn(F.relu(self.conv6(x))))
        x = self.dropout40(x)
        
        # flatten output of third convolutional layer into a vector
        # this vector is passed through the fully-connected nn
        x = x.view(-1, 128 * 4 * 4)
        # add dropout layer
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout50(x)
        # add 2nd hidden layer, without relu activation function
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


# -------------------------
# Model definition
# -------------------------
class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def cifar10_vgg11_bn() -> VGG:
    return VGG(make_layers(cfgs['A'], batch_norm=True), num_classes=10)