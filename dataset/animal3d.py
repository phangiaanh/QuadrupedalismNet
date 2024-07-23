from .base_dataset import BaseDataset, base_loader
from glob import glob
import os
import json

class AnnotationItem():

    def __init__(self, 
        img_path:str, mask_path:str, rd_mask_path:str,
        bbox:list,
        keypoint_2d:list, reproj_kp_2d:list, joints_2d:list, 
        keypoint_3d:list, joints_3d:list, 
        pose:list, shape:list, shape_extra:list, trans:list,
        ):
        pass

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
        self.num_images = len(images)
        print(self.num_images)

        # map[AnnotationItem]
        self.annotations = self.load_annotations(images)
        # for i, img  in enumerate(images):
        #     print(img)


    def load_annotations(self, anno_paths):
        self.annotations = {}

        anno_file = os.path.join(self.data_dir, self.folder_name + '.json')
        

        # for i, img_path in enumerate(anno_paths):



    def __len__(self):
        return self.num_images
    

    



def data_loader(cfg):
    return base_loader(Animal3DDataset, cfg, filter_keys=None)



