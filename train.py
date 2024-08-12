from trainer.shape_trainer import ShapeTrainer
from config import cfg
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__=="__main__":

    # if not torch.cuda.is_available():
    #     raise Exception("Cuda Not Implemented")

    trainer = ShapeTrainer(cfg)
    trainer.init_training()
    trainer.train()
