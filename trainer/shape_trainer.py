import pickle
import torch
import torchvision
import os
import numpy as np
import time
from models.quadrupedalismnet import QuadrupedalismNet
from trainer.basic_trainer import Trainer
from utils.criterions import *
from utils.image import compute_dt_barrier

class ShapeTrainer(Trainer):

  def define_model(self):
    cfg = self.cfg
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.save_directory = cfg['CHECKPOINT_DIR']
    self.image_size = (cfg['MODEL']['IMAGE_SIZE'][0], cfg['MODEL']['IMAGE_SIZE'][1])

    self.data = pickle.load(open(cfg['DATA_PATH'], 'rb'), encoding='latin1')
    eigenvalues, eigenvectors = np.linalg.eig(self.data['toys_betas'])

    pca_var = eigenvalues[:cfg['TRAIN']['NUM_BETAS']]
    #self.betas_precision = torch.Tensor(pca_var).cuda().expand(cfg['TRAIN']['BATCH_SIZE_PER_GPU'], cfg['TRAIN']['NUM_BETAS'])


    self.model = QuadrupedalismNet(self.image_size, self.cfg)
    if cfg['TRAIN']['BEGIN_EPOCH'] > 0:
      self.load_network(self.model, "quadrupedalism")

    if torch.cuda.is_available():
      self.model = self.model.cuda()
    
    edges2verts = self.model.edges2verts
    edges2verts = np.tile(np.expand_dims(edges2verts, 0), (self.cfg['TRAIN']['BATCH_SIZE_PER_GPU'], 1, 1))

    self.resnet_transform = torchvision.transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    self.invalid_batch = False


  def define_criterion(self):
    self.projection_loss = kp_12_loss
    self.mask_loss = mask_loss
    
    self.model_trans_loss_fn = model_trans_loss
    self.model_pose_loss_fn = model_pose_loss
    
  
  def set_input(self, batch):
    img_tensor = batch['img'].clone().type(torch.FloatTensor)
    print(img_tensor.shape)

    input_img_tensor = batch['img'].type(torch.FloatTensor)

    for b in range(input_img_tensor.size(0)):
      input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

    if torch.cuda.is_available():
      self.input_imgs = input_img_tensor.cuda()
      self.imgs = img_tensor.cuda()
    else:
      self.input_imgs = input_img_tensor
      self.imgs = img_tensor

    if 'mask' in batch.keys():
      mask_tensor = batch['mask'].type(torch.FloatTensor)
      if torch.cuda.is_available():
        self.masks = mask_tensor.cuda()
      else:
        self.masks = mask_tensor

    # print(batch['keypoints_2d'])
    # print(len(batch['keypoints_2d']))
    # print(len(batch['keypoints_2d'][0]))
    # print(len(batch['keypoints_2d'][0][0]))
    if 'keypoints_2d' in batch.keys():
      # print(batch['keypoints_2d'])
      # kp_tensor = torch.FloatTensor(batch['keypoints_2d'])
      # kp_tensor = torch.stack(batch['keypoints_2d'])
      kp_tensor = batch['keypoints_2d'].type(torch.FloatTensor)
      # print('aaa')
      # print(kp_tensor.shape)
      if torch.cuda.is_available():
        self.kps2 = kp_tensor.cuda()
      else:
        self.kps2 = kp_tensor

    if 'pose' in batch.keys():
      model_pose_tensor = batch['pose'].type(torch.FloatTensor)
      if torch.cuda.is_available():
        self.model_pose = model_pose_tensor.cuda()
      else:
        self.model_pose = model_pose_tensor

    if 'shape' in batch.keys():
      model_shape_tensor = batch['shape'].type(torch.FloatTensor)
      if torch.cuda.is_available():
        self.model_shape = model_shape_tensor.cuda()
      else:
        self.model_shape = model_shape_tensor

    if 'trans' in batch.keys():
      model_trans_tensor = batch['trans'].type(torch.FloatTensor)
      if torch.cuda.is_available():
        self.model_trans = model_trans_tensor.cuda()
      else:
        self.model_trans = model_trans_tensor

    if self.masks is not None:
      mask_dts = np.stack([compute_dt_barrier(m) for m in batch['mask']])
      dt_tensor = torch.FloatTensor(mask_dts)

      if torch.cuda.is_available():
        self.dts_barrier = dt_tensor.cuda().unsqueeze(1)
      else:
        self.dts_barrier = dt_tensor.unsqueeze(1)  

  def forward(self):

    pass
    
