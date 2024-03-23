import torch.nn as nn


BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):  
    out = self.conv(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)  
    residual = x
    if self.downsample is not None:
      residual = self.downsample(x)  
    out += residual
    out = self.relu(out)  
    return out

class BottleneckBlock(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BottleneckBlock, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):  
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)  
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)  
    out = self.conv3(out)
    out = self.bn3(out)  
    residual = x
    if self.downsample is not None:
      residual = self.downsample(x)  
    out += residual
    out = self.relu(out)  
    return out