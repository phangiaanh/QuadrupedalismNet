import torch
import torch.nn as nn
import numpy as np
from . import net_blocks as nb

class ShapePredictor(nn.Module):
  def __init__(self, nz_feat, num_verts, cfg, left_idx, right_idx, shapedirs):
    super(ShapePredictor, self).__init__()
    self.cfg = cfg
    B = shapedirs
    self.pred_layer = nn.Linear(nz_feat, num_verts * 3)
    self.fc = nb.fc('batch', nz_feat, self.cfg['MODEL']['N_SHAPE_FEAT'])
    n_feat = self.cfg['MODEL']['N_SHAPE_FEAT']
    B = B.permute(1,0)
    A = torch.Tensor(np.zeros((B.size(0), n_feat)))
    n = np.min((B.size(1), n_feat))
    A[:,:n] = B[:,:n]
    self.pred_layer.weight.data = torch.nn.Parameter(A)
    self.pred_layer.bias.data.fill_(0.)

  def forward(self, feat):
    delta_v = self.pred_layer.forward(feat)
            # Make it B x num_verts x 3
    delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
    return delta_v

class ScalePredictor(nn.Module):
  def __init__(self, nz_feat, cfg):
    super(ScalePredictor, self).__init__()
    self.cfg = cfg
    self.pred_layer = nn.Linear(nz_feat, 1)

  def forward(self, feat):
    scale = self.pred_layer.forward(feat)   
    return scale

class TransPredictor(nn.Module):
  def __init__(self, nz_feat, cfg):
    super(TransPredictor, self).__init__()
    self.cfg = cfg
    self.pred_layer_xy = nn.Linear(nz_feat, 2)
    self.pred_layer_z = nn.Linear(nz_feat, 1)

    self.pred_layer_xy.weight.data.normal_(0, 0.0001)
    self.pred_layer_xy.bias.data.normal_(0, 0.0001)
    
    self.pred_layer_z.weight.data.normal_(0, 0.0001)
    self.pred_layer_z.bias.data.normal_(0, 0.0001)

  def forward(self, feat):
    if torch.cuda.is_available():
      trans = torch.Tensor(np.zeros((feat.shape[0],3))).cuda()
      f = torch.Tensor(np.zeros((feat.shape[0],1))).cuda()
    else:
      trans = torch.Tensor(np.zeros((feat.shape[0],3)))
      f = torch.Tensor(np.zeros((feat.shape[0],1)))
      
    feat_xy = feat
    feat_z = feat
    trans[:,:2] = self.pred_layer_xy(feat_xy)
    trans[:,0] += 1.0
    trans[:,2] = 1.0+self.pred_layer_z(feat_z)[:,0]

    return trans

class PosePredictor(nn.Module):
  def __init__(self, nz_feat, cfg, num_joints=35):
    super(PosePredictor, self).__init__()
    self.cfg = cfg
    self.num_joints = num_joints
    self.pred_layer = nn.Linear(nz_feat, num_joints*3)

  def forward(self, feat):
    pose = self.pred_layer.forward(feat)

    # Add this to have zero to correspond to frontal facing
    pose[:,0] += 1.20919958
    pose[:,1] += 1.20919958
    pose[:,2] += -1.20919958
    return pose


class Predictor(nn.Module):
  def __init__(self, nz_feat=100, nenc_feat=2048, num_verts=1000, cfg=None, left_idx=None, right_idx=None, shapedirs=None):
    super(Predictor, self).__init__()
    self.cfg = cfg
    self.shape_predictor = ShapePredictor(nz_feat=nz_feat, num_verts=num_verts, cfg=self.cfg, left_idx=left_idx, right_idx=right_idx, shapedirs=shapedirs)
    self.scale_predictor = ScalePredictor(nz_feat=nz_feat, cfg=self.cfg)
    self.trans_predictor = TransPredictor(nz_feat=nz_feat, cfg=self.cfg)
    self.pose_predictor = PosePredictor(nz_feat=nz_feat, cfg=self.cfg)


  def forward(self, feat, enc_feat):
    shape_pred = self.shape_predictor.forward(feat)
    scale_pred = self.scale_predictor.forward(feat)
    trans_pred = self.trans_predictor.forward(feat)
    pose_pred = self.pose_predictor.forward(feat)

    return shape_pred, scale_pred, trans_pred, pose_pred


