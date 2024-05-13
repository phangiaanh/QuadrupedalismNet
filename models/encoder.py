import torch.nn as nn
from .resnet50 import ResNetConv
from . import net_blocks as nb

class Encoder(nn.Module):
  def __init__(self, cfg, input_shape, n_blocks=4, nz_feat=100, bott_size=256):
    super(Encoder, self).__init__()
    self.cfg = cfg
    self.resnet_conv = ResNetConv(n_blocks=4, opts=cfg)
    num_norm_groups = bott_size//self.cfg['MODEL']['CHANNEL_PER_GROUP']
    
    self.enc_conv1 = nb.conv2d('group', 2048, bott_size, stride=2, kernel_size=4, num_groups=num_norm_groups)
    nc_input = bott_size * (input_shape[0] // 64) * (input_shape[1] // 64)
    self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2, 'batch')
    self.nenc_feat = nc_input
    nb.net_init(self.enc_conv1)

  def forward(self, img, fg_img):
    resnet_feat = self.resnet_conv.forward(img, fg_img)
    out_enc_conv1 = self.enc_conv1(resnet_feat)
    out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
    feat = self.enc_fc.forward(out_enc_conv1)
    return feat, out_enc_conv1