
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from glob import glob

class BaseDataset(Dataset):

    def __init__(self, cfg, filter_keys=None):
        self.cfg = cfg
        self.img_size = self.cfg['MODEL']['IMAGE_SIZE']
        self.filter_keys = filter_keys


        with open(self.cfg['MODEL_PATH'], 'rb') as f:
            dd = pickle.load(f, encoding='latin1')
            num_betas = dd['shapedirs'].shape[-1]
            self.shapedirs = np.reshape(dd['shapedirs'], [-1, num_betas]).T
        
        print(self.shapedirs.shape)

    def __len__(self):
        return 10



def base_loader(dataset_init, cfg, filter_keys=None):
    dataset = dataset_init(cfg, filter_keys=filter_keys)
    return DataLoader(
        dataset,
        batch_size=cfg['TRAIN']['BATCH_SIZE_PER_GPU'],
        shuffle=cfg['TRAIN']['SHUFFLE'],
        num_workers=cfg['TRAIN']['N_DATA_WORKERS'],
        drop_last=True
    )