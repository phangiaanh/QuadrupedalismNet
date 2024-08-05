from config import cfg
from dataset.animal3d import AnnotationItem
import os
import json
import argparse
from matplotlib import image, patches
from matplotlib import pyplot

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

    def show_annotation(self, key):
        if key in self.annotations:
            print(f"{key} existed")
        else:
            print(f"{key} does not exist")

        return self.annotations[key]
    
    def visualize_annotation(self, key):

        annotation = self.show_annotation(key=key)
        print(annotation)

        base_img = image.imread(os.path.join(self.data_dir, annotation.img_path))
        fig, ax = pyplot.subplots()
        rec = patches.Rectangle((annotation.bbox[0], annotation.bbox[1]), annotation.bbox[2] - annotation.bbox[0], annotation.bbox[3] - annotation.bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        pyplot.imshow(base_img)
        ax.add_patch(rec)
        ax.scatter([x[0] for x in annotation.keypoint_2d], [x[1] for x in annotation.keypoint_2d])
        pyplot.savefig("test/visualize.png")



if __name__=="__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--path', metavar='N', type=str, nargs='+', help='a path for visualizing keypoints')

    args = parser.parse_args()
    path = args.path[0]

    animal3DAnno = Animal3DLoader(cfg=cfg)

    animal3DAnno.visualize_annotation(path)


