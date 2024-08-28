from .base_dataset import BaseDataset, base_loader
from glob import glob
import os
import json
from PIL import Image
import torch
from torchvision import transforms

class AnnotationItem():

    def __init__(self, 
        img_path:str, mask_path:str, rd_mask_path:str,
        bbox:list,
        keypoint_2d:list, reproj_kp_2d:list, joints_2d:list, 
        keypoint_3d:list, joints_3d:list, 
        pose:list, shape:list, shape_extra:list, trans:list,
        supercategory:int, category:int, with_tail:int, height:int, width:int
        ):
        self.img_path = img_path
        self.mask_path = mask_path
        self.rd_mask_path = rd_mask_path
        self.bbox = bbox
        self.keypoint_2d = keypoint_2d
        self.reproj_kp_2d = reproj_kp_2d
        self.joints_2d = joints_2d
        self.keypoint_3d = keypoint_3d
        self.joints_3d = joints_3d
        self.pose = pose
        self.shape = shape
        self.shape_extra = shape_extra
        self.trans = trans
        self.supercategory = supercategory
        self.category = category
        self.with_tail = with_tail
        self.height = height
        self.width = width

    def __repr__(self) -> str:
        
        return f"""************************
Image Path   : {self.img_path}
Mask Path    : {self.mask_path}
Bounding Box : {self.bbox}
2D Keypoints : {self.keypoint_2d}
2D Joints    : {self.joints_2d}
3D Keypoints : {len(self.keypoint_3d)}
3D Joints    : {len(self.joints_3d)}
Pose         : {len(self.pose)}
Shape        : {len(self.shape)}
Trans        : {self.trans}
Height       : {self.height}
Width        : {self.width}
************************"""
        

class Animal3DDataset(BaseDataset):

    def __init__(self, cfg, is_inference=True, filter_keys=None):
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
        self.load_annotations(images)
        self.transform = transforms.Compose([
            transforms.Resize((self.cfg['MODEL']['IMAGE_SIZE'][0], self.cfg['MODEL']['IMAGE_SIZE'][1])),  
            transforms.ToTensor(),  
        ])
        # for i, img  in enumerate(images):
        #     print(img)


    def load_annotations(self, anno_paths):
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
        self.annotations_list:list[AnnotationItem] = list([self.annotations[key] for key in self.annotations])

        # loop check
        counter = 0
        for i, img_path in enumerate(anno_paths):
            path_items = img_path.split('/')
            img_key = os.path.join(*path_items[-4:])
            if img_key not in self.annotations: 
                print(f"Something wrong with key {img_key}")
                continue
            counter += 1
        
        if counter != len(self.annotations):
            print("Missing images")
        else:
            print("All images annotated")



    def __len__(self):
        print(f"OOOOOOOOOOOOOO: {len(self.annotations_list)}")
        return len(self.annotations_list)
    

    def forward_img(self, index):
        anno_data = self.annotations_list[index]
        img = Image.open(os.path.join(self.data_dir, anno_data.img_path)).convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, anno_data.mask_path)).convert("RGB")

        
        return img, mask, anno_data.joints_2d, anno_data.joints_3d, anno_data.keypoint_2d, anno_data.keypoint_3d, anno_data.pose, anno_data.shape, anno_data.trans

    def __getitem__(self, index):
        img, mask, j2, j3, k2, k3, sp, ss, st = self.forward_img(index)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        k2 = torch.Tensor(k2)
        print(f'index: {index}')
        print(f'k2: {k2}')
        # print(f'k2: {len(k2)}')
        # print(f'k2: {len(k2[0])}')
        # print(f'k2: {len(k2[0][0])}')
        
        elem = {
            'img': img,
            'mask': mask,
            'joints_2d': j2,
            'joints_3d': j3,
            'keypoints_2d': k2,
            'keypoints_3d': k3,
            'pose': sp,
            'shape': ss,
            'trans': st,
            # 'anno': anno

        }
        return elem
































def data_loader(cfg):
    return base_loader(Animal3DDataset, cfg, filter_keys=None)



