from .base_dataset import BaseDataset, base_loader
from glob import glob
import os

class Animal3DDataset(BaseDataset):

    def __init__(self, cfg, is_inference=False, filter_keys=None):
        super(Animal3DDataset, self).__init__(cfg, filter_keys)
        self.cfg = cfg
        self.filter_keys = filter_keys
        if is_inference:
            self.folder_name = 'test'
        else:
            self.folder_name = 'train'

        self.data_dir = self.cfg['DATASET']['DATA_DIR']
        self.img_dir = os.path.join(self.data_dir, 'images')
        print(self.img_dir)
        pattern1 = os.path.join(self.img_dir, self.folder_name, '*', '*.JPEG') 
        pattern2 = os.path.join(self.img_dir, self.folder_name, '*', '*.jpg')
        images = glob(pattern1, recursive=True) + glob(pattern2, recursive=True)
        num_images = len(images)
        print(num_images)

    



def data_loader(cfg):
    return base_loader(Animal3DDataset, cfg, filter_keys=None)
