import torch.nn as nn
from .base import BN_MOMENTUM

class ResNet50(nn.Module):

  def __init__(self, block, layers, smal_mean_params, **kwargs):
    self.inplanes = 64
    super(ResNet50, self).__init__()
    npose = 24 * 6

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # 4 ResNet Layers
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    # FC Layers
    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
    self.drop1 = nn.Dropout()
    self.fc2 = nn.Linear(1024, 1024)
    self.drop2 = nn.Dropout()




  def _make_layer(self, block, planes, blocks, stride=1):
    downsample=None
    if (stride != 1) or (self.inplanes != planes * block.expansion):
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

