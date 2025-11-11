import os
from tqdm import tqdm
from beartype.typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist
from pytorch_lightning import LightningDataModule

from . import utils
            
class OmniDataModule(LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, predict_dataset=None,
                 batch_size=16, num_workers=4, sampler=None, collater=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False, 
            ignore=['train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset', 'sampler', 'collater'])
        
        self.is_child_process = utils.is_child_process()
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.sampler = sampler # not used yet
        self.collater = collater # not used yet

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cfg = kwargs


    def train_dataloader(self, rank=None, num_replicas=None):
        if self.train_dataset is None:
            return None
        return DataLoader(self.train_dataset, 
                          shuffle=self.cfg.get('train_shuffle', True),
                          batch_size=self.cfg.get('train_batch_size', self.batch_size),
                          num_workers=self.cfg.get('train_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('train_prefetch_factor', 2),
                          pin_memory=self.cfg.get('train_pin_memory', True),
                          persistent_workers=self.cfg.get('train_persistent_workers', True),
                          )
    def val_dataloader(self):
        # val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, 
                          shuffle=self.cfg.get('val_shuffle', False),
                          batch_size=self.cfg.get('val_batch_size', self.batch_size),
                          num_workers=self.cfg.get('val_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('val_prefetch_factor', 2),
                          pin_memory=self.cfg.get('val_pin_memory', True),
                          persistent_workers=self.cfg.get('val_persistent_workers', True),
                          )
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, 
                          shuffle=self.cfg.get('test_shuffle', False),
                          batch_size=self.cfg.get('test_batch_size', self.batch_size),
                          num_workers=self.cfg.get('test_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('test_prefetch_factor', 2),
                          pin_memory=self.cfg.get('test_pin_memory', False),
                          persistent_workers=self.cfg.get('test_persistent_workers', True),
                          )
    def predict_dataloader(self):
        if self.predict_dataset is None:
            return None
        return DataLoader(self.predict_dataset, 
                          shuffle=self.cfg.get('predict_shuffle', False),
                          batch_size=self.cfg.get('predict_batch_size', self.batch_size),
                          num_workers=self.cfg.get('predict_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('predict_prefetch_factor', 2),
                          pin_memory=self.cfg.get('predict_pin_memory', False),
                          persistent_workers=self.cfg.get('predict_persistent_workers', True),
                          )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any] = None):
        """Things to do when loading checkpoint."""
        pass
