import os
import math
from beartype.typing import Any, Dict, Optional

import torch
from torch.utils.data.distributed import DistributedSampler, dist
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

class OmniDataModule(LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, predict_dataset=None,
                 batch_size=16, num_workers=4, sampler=None, collater=None,
                 **kwargs):
        super().__init__()
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = sampler # not used yet
        self.collater = collater # not used yet

        self.cfg = kwargs
        self.save_hyperparameters(logger=False, ignore=['train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset', 'sampler', 'collater'])


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
        if self.val_dataset is None:
            return None
        # val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
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

"""
Taken from
https://github.com/microsoft/protein-frame-flow/blob/main/data/pdb_dataloader.py#L162
"""
class RNALengthBatcher:
    def __init__(
            self,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple RNA of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_na_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)
        
        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947

        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        if hasattr(self, "sample_order"):
            return len(self.sample_order)
        else:
            return self._num_batches
        

# class MyBatchCollator:
#     """ batch_dim: whether batch dim already exists in the input 
#         set_length: the length of the set, if None, use the max length in the batch
#         multiple_of: if set_length is None, use the multiple of this number as set_length
#     """
#     def __init__(self, batch_dim=True, set_length=None, multiple_of=64) -> None:
#         """ batch_dim: whether batch dim already exists in the input """
#         self.batch_dim = batch_dim
#         self.set_length = set_length
#         self.multiple_of = multiple_of
#         if set_length is not None and multiple_of is not None:
#             logger.warning(f"set_length and multiple_of are both specified, multiple_of will be ignored")
    
#     def __call__(self, samples):

#         if self.set_length is not None or self.multiple_of is not None or len(samples) > 1:
#             samples = tailor_data_shapes(samples,
#                                    set_length=self.set_length, 
#                                    multiple_of=self.multiple_of, 
#                                    batch_dim=self.batch_dim)

#         if self.batch_dim:
#             stack_fn = partial(torch.concat, dim=0)
#         else:
#             stack_fn = partial(torch.stack, dim=0)

#         return dicts_aggregate(stack_fn, samples)


class MyDataLoader(DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        # this block is present in opencomplex code but not in openfold code
        # if(stage_cfg.supervised):
        #     clamp_prob = self.config.supervised.clamp_prob
        #     keyed_probs.append(
        #         ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
        #     )

        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)
