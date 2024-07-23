from yacs.config import CfgNode as CN

cfg = CN()
cfg.OUTPUT_DIR = 'output'
cfg.LOG_DIR = ''
cfg.DATA_DIR = ''
cfg.CHECKPOINT_DIR = 'content'
cfg.GPUS = (0, 1)
cfg.WORKERS = 4
cfg.PRINT_FREQ = 100
cfg.AUTO_RESUME = True
cfg.LABEL_PER_CLASS = 15
cfg.MODEL_PATH = 'SMAL/smal_CVPR2018.pkl'
cfg.DATA_PATH = 'SMAL/smal_CVPR2018_data.pkl'
cfg.GEN_DIR = 'SMAL'

# CUDNN params
cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# MODEL params
cfg.MODEL = CN()
cfg.MODEL.NAME = 'pose_hrnet'
cfg.MODEL.INIT_WEIGHTS = True
cfg.MODEL.PRETRAINED = 'models/pytorch/imagenet/hrnet_w32-36af842e.pth'
cfg.MODEL.NUM_JOINTS = 17
cfg.MODEL.TAG_PER_JOINT = True
cfg.MODEL.TARGET_TYPE = 'gaussian'
cfg.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
cfg.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
cfg.MODEL.SIGMA = 2
cfg.MODEL.NZ_FEAT = 1024
cfg.MODEL.BOTTLENECK_SIZE = 2048
cfg.MODEL.CHANNEL_PER_GROUP = 16
cfg.MODEL.NUM_KPS = 28
cfg.MODEL.N_SHAPE_FEAT = 40
cfg.MODEL.OPTIMIZATION = False
cfg.MODEL.EXTRA = CN(new_allowed=True)

cfg.MODEL.EXTRA.PRETRAINED_LAYERS = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3', 'transition3', 'stage4']
cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

# STAGE2 params
cfg.MODEL.EXTRA.STAGE2 = CN()
cfg.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
cfg.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
cfg.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
cfg.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
cfg.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

# STAGE3 params
cfg.MODEL.EXTRA.STAGE3 = CN()
cfg.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
cfg.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
cfg.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
cfg.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
cfg.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
cfg.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

# STAGE4 params
cfg.MODEL.EXTRA.STAGE4 = CN()
cfg.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
cfg.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
cfg.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
cfg.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
cfg.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

# LOSS params
cfg.LOSS = CN()
cfg.LOSS.USE_TARGET_WEIGHT = True

# TRAIN params
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE_PER_GPU = 32
cfg.TRAIN.SHUFFLE = True
cfg.TRAIN.BEGIN_EPOCH = 0
cfg.TRAIN.END_EPOCH = 210
cfg.TRAIN.OPTIMIZER = 'adam'
cfg.TRAIN.LR = 0.001
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.LR_STEP = [170, 200]
cfg.TRAIN.WD = 0.0001
cfg.TRAIN.GAMMA1 = 0.99
cfg.TRAIN.GAMMA2 = 0
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.NUM_BETAS = 20
cfg.TRAIN.N_DATA_WORKERS = 4

# DATASET params
cfg.DATASET = CN()
cfg.DATASET.COLOR_RGB = True
cfg.DATASET.DATA_DIR = '/home/watermelon/Thesis/animal3d'
cfg.DATASET.DATASET = 'animal3d'
cfg.DATASET.DATA_FORMAT = 'jpg'
cfg.DATASET.FLIP = True
cfg.DATASET.NUM_JOINTS_HALF_BODY = 8
cfg.DATASET.PROB_HALF_BODY = 0.3
cfg.DATASET.ROOT = 'data/animalpose/'
cfg.DATASET.ROT_FACTOR = 40
cfg.DATASET.SCALE_FACTOR = 0.5
cfg.DATASET.TEST_SET = 'test'
cfg.DATASET.TRAIN_SET = 'train'
cfg.DATASET.VAL_SET = 'val'
cfg.DATASET.SELECT_DATA = False
cfg.DATASET.SUPERCATEGORY = ['Bovidae']
cfg.DATASET.SMAL_MEAN_PARAMS = 'data/smpl_mean_params.npz'

# TEST params
cfg.TEST = CN()
cfg.TEST.BATCH_SIZE_PER_GPU = 32
cfg.TEST.COCO_BBOX_FILE = ''
cfg.TEST.BBOX_THRE = 1
cfg.TEST.IMAGE_THRE = 0
cfg.TEST.IN_VIS_THRE = 0.2
cfg.TEST.MODEL_FILE = ''
cfg.TEST.NMS_THRE = 1
cfg.TEST.OKS_THRE = 0.9
cfg.TEST.USE_GT_BBOX = True
cfg.TEST.FLIP_TEST = True
cfg.TEST.POST_PROCESS = True
cfg.TEST.SHIFT_HEATMAP = True

# DEBUG params
cfg.DEBUG = CN()
cfg.DEBUG.DEBUG = True
cfg.DEBUG.SAVE_BATCH_IMAGES_GT = True
cfg.DEBUG.SAVE_BATCH_IMAGES_PRED = True
cfg.DEBUG.SAVE_HEATMAPS_GT = True
cfg.DEBUG.SAVE_HEATMAPS_PRED = True