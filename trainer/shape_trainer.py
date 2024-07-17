import pickle
import torch
import os
import numpy as np
from models.quadrupedalismnet import QuadrupedalismNet
from trainer.basic_trainer import Trainer
from utils.criterions import *

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

  def define_criterion(self):
    self.projection_loss = kp_12_loss
    self.mask_loss = mask_loss
    
    self.model_trans_loss_fn = model_trans_loss
    self.model_pose_loss_fn = model_pose_loss

    

