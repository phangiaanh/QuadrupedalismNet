from .base_dataset import BaseDataset, base_loader

class Animal3DDataset(BaseDataset):

    def __init__(self, cfg, filter_keys=None):
        super(Animal3DDataset, self).__init__(cfg, filter_keys)
        self.filter_keys = filter_keys

        

    



def data_loader(cfg):
    return base_loader(Animal3DDataset, cfg, filter_keys=None)
