
from .smal import SMAL
import torch
import torch.nn as nn
import numpy as np
from .poseverts import compute_edges2verts
from .encoder import Encoder
from .predictor import Predictor

class QuadrupedalismNet(nn.Module):

  def __init__(self, input_shape, cfg):
    super(QuadrupedalismNet, self).__init__()
    self.cfg = cfg
    self.smal = SMAL(self.cfg['MODEL_PATH'], cfg)

    self.op_features = None

    self.left_idx = np.hstack((self.smal.left_inds, self.smal.center_inds))
    self.right_idx = np.hstack((self.smal.right_inds, self.smal.center_inds))

    pose = np.zeros((1,105))
    betas = np.zeros((1,self.cfg["TRAIN"]["NUM_BETAS"]))
    V,J,R = self.smal(torch.Tensor(betas), torch.Tensor(pose))
    verts = V[0,:,:]
    verts = verts.data.cpu().numpy()
    faces = self.smal.f

    num_verts = verts.shape[0]
    self.mean_v = nn.Parameter(torch.Tensor(verts))
    self.num_output = num_verts
    faces = faces.astype(np.int32) 

    verts_np = verts
    faces_np = faces

    if torch.cuda.is_available():
      self.faces = torch.LongTensor(faces).cuda()
    else:
      self.faces = torch.LongTensor(faces)
      
    self.edges2verts = compute_edges2verts(verts, faces)

    vert2kp_init = torch.Tensor(np.ones((self.cfg['MODEL']['NUM_KPS'], num_verts)) / float(num_verts))
    # Remember initial vert2kp (after softmax)
    if torch.cuda.is_available():
      self.vert2kp_init = torch.nn.functional.softmax(vert2kp_init.cuda(), dim=1)
    else:
      self.vert2kp_init = torch.nn.functional.softmax(vert2kp_init, dim=1)
    self.vert2kp = nn.Parameter(vert2kp_init)


    self.encoder = Encoder(self.cfg, input_shape, n_blocks=4, nz_feat=self.cfg['MODEL']['NZ_FEAT'], bott_size=self.cfg['MODEL']['BOTTLENECK_SIZE'])
    nenc_feat = self.encoder.nenc_feat
    self.code_predictor = Predictor(nz_feat=self.cfg['MODEL']['NZ_FEAT'], nenc_feat=nenc_feat, num_verts=self.num_output, cfg=self.cfg, left_idx=self.left_idx, right_idx=self.right_idx, shapedirs=self.smal.shapedirs)

  
  def forward(self, img, masks):
    if self.cfg['MODEL']['OPTIMIZATION']:
      if self.op_features is None:
        img_feat, enc_feat = self.encoder.forward(img, masks)
        self.op_features = Variable(img_feat.cuda(device=self.opts.gpu_id), requires_grad=True)
      codes_pred = self.code_predictor.forward(self.op_features, None)
      img_feat = self.op_features
    else:
      img_feat, enc_feat = self.encoder.forward(img, masks)
      codes_pred = self.code_predictor.forward(img_feat, enc_feat)

    return codes_pred