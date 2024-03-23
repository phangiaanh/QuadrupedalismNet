from models.shape_trainer import Trainer
from config import cfg

import sys
sys.path.append('models')

if __name__=="__main__":
    trainer = Trainer(cfg)