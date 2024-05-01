
from .smal import SMAL
import torch
import torch.nn as nn
import numpy as np

class QuadrupedalismNet(nn.Module):

  def __init__(self, input_shape, cfg):
    super(QuadrupedalismNet, self).__init__()
    self.cfg = cfg
    self.smal = SMAL(self.cfg['MODEL_PATH'], cfg)

    pose = np.zeros((1,99))
    betas = np.zeros((1,self.cfg["TRAIN"]["NUM_BETAS"]))
    V,J,R = self.smal(torch.Tensor(betas), torch.Tensor(pose))
    



