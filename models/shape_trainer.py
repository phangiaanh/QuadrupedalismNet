import pickle
import torch
import os
import numpy as np
from .quadrupedalismnet import QuadrupedalismNet

class Trainer():

  def __init__(self, cfg):
    self.cfg = cfg
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

    #self.model = self.model.cuda()



  def load_network(self, network, label, epoch, network_directory=None):
    save_filename = f"{label}_{epoch}.pth"
    if network_directory is None:
        network_directory = self.save_directory

    save_path = os.path.join(network_directory, save_filename)
    return network.load_state_dict(torch.load(save_path))

  def save_network(self, network, label, epoch, network_directory= None):
    save_filename = f"{label}_{epoch}.pth"
    if network_directory is None:
        network_directory = self.save_directory

    save_path = os.path.join(network_directory, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
      network.cuda()
    return

