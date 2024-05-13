import torch.nn as nn
from .base import BN_MOMENTUM
import torchvision
from torchvision.models import ResNet50_Weights

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

class ResNetConv(nn.Module):
  def __init__(self, n_blocks=4, opts=None):
    super(ResNetConv, self).__init__()
    self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    self.n_blocks = n_blocks
    self.opts = opts
    # if self.opts.use_double_input:
    #     self.fc = nb.fc_stack(512*16*8, 512*8*8, 2)
  def forward(self, x, y=None):
    # if self.opts.use_double_input and y is not None:
    #     x = torch.cat([x, y], 2)
    n_blocks = self.n_blocks
    x = self.resnet.conv1(x)
    x = self.resnet.bn1(x)
    x = self.resnet.relu(x)
    x = self.resnet.maxpool(x)
    if n_blocks >= 1:
        x = self.resnet.layer1(x)
    if n_blocks >= 2:
        x = self.resnet.layer2(x)
    if n_blocks >= 3:
        x = self.resnet.layer3(x)
    if n_blocks >= 4:
        x = self.resnet.layer4(x)
    # if self.opts.use_double_input and y is not None:
    #     x = x.view(x.size(0), -1)
    #     x = self.fc.forward(x)
    #     x = x.view(x.size(0), 512, 8, 8)
        
    return 