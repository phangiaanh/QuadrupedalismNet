from config import cfg
from dataset.animal3d import AnnotationItem
import os
import json
import argparse

class Animal3DLoader():

    def __init__(self, cfg, is_inference=True, filter_keys=None):
        self.cfg = cfg
        self.filter_keys = filter_keys
        if is_inference:
            self.folder_name = 'test'
        else:
            self.folder_name = 'train'

        self.data_dir = self.cfg['DATASET']['DATA_DIR']
        self.img_dir = os.path.join(self.data_dir, 'images')

        # map[AnnotationItem]
        self.load_annotations()
        


    def load_annotations(self):
        self.annotations = {}

        anno_file = os.path.join(self.data_dir, self.folder_name + '.json')
        
        if not os.path.isfile(anno_file):
            print(f"No exist file config cam.json {anno_file}")
            return
        else:
            with open(anno_file, 'r') as json_file:
                data = json.load(json_file)

        anno_data = data["data"]
        for anno in anno_data:
            item = AnnotationItem(**anno)
            self.annotations[item.img_path] = item
        self.annotations_list = list([self.annotations[key] for key in self.annotations])


if __name__=="__main__":
