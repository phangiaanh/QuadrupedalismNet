import pickle
import torch.nn as nn

class SMAL(object):
  def __init__(self, model_path, cfg):
    self.cfg = cfg

    dd = pickle.load(open(model_path, 'rb'), encoding='latin1')




class QuadrupedalismNet(nn.Module):

  def __init__(self, input_shape, cfg):
    super(QuadrupedalismNet, self).__init__()
    self.cfg = cfg
    self.smal = SMAL(self.cfg['MODEL_PATH'], cfg)




