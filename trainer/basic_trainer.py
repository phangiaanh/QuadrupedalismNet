
import sys
# sys.path.append("../")

import os
import torch
from  dataset.animal3d import data_loader


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_dir = os.path.join(cfg['CHECKPOINT_DIR'], "QuadrupedalismNet")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        log_file = os.path.join(self.save_dir, 'config.yaml')
        with open(log_file, 'w') as f:
            for k in (self.cfg):
                # print(k)
                f.write('{}: {}\n'.format(k, self.cfg.__getattr__(k)))

        pass

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

    def define_model(self):
        raise NotImplementedError
    
    def define_criterion(self):
        raise NotImplementedError
    
    def init_dataset(self):
        self.data_loader = data_loader(self.cfg)

    def init_training(self):
        self.init_dataset()
        self.define_model()
        self.define_criterion()