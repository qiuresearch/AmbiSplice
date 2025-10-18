import os
import h5py
import tables
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from . import splice_feats
from . import utils

ilogger = utils.get_pylogger(__name__)


class PangolinDataset(Dataset):
    """ Adapted from Pangolin github repo """
    def __init__(self, file_path, **kwargs):
        super(PangolinDataset, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        self.ctr = 0

    def __getitem__(self, idx):
        self.ctr += 1
        if self.ctr % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.file_path, 'r', libver='latest')
        X = self.data['X' + str(idx)][:].T # transpose to (4, L)
        Y = self.data['Y' + str(idx)][:].T # transpose to (12, L)
        Z = self.data['Z' + str(idx)][:] # (4,): [chrom, start, end, strand]

        # tissue order: heart, liver, brain, testis
        # each tissue: unspliced, spliced, usage
        sample_feats = {
            'seq': splice_feats.decode_onehot(X, dim=0, idx2base=np.array(['A', 'C', 'G', 'T', 'N'])),
            'seq_onehot': X,
            'cls': np.argmax(Y[0:2, :], axis=0), # 0: unspliced, 1: spliced
            'psi': Y[2, :], # usage
            'chrom': Z[0].decode(),
            'start': int(Z[1]),
            'end': int(Z[2]),
            'strand': Z[3].decode(),
        }

        return sample_feats

    def __len__(self):
        assert len(self.data) % 3 == 0
        return len(self.data) // 3
    
    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in PangolinDataset.")


class GeneCropsDataset(Dataset):
    def __init__(self, data, file_path=None, indices=None, num_classes=None, **kwargs):
        super(GeneCropsDataset, self).__init__()

        self.data = data
        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise FileNotFoundError(f"Data file {self.data} does not exist.")
            self.file_path = self.data
            self.data = tables.open_file(self.file_path, mode='r')
        elif self.data is None: # load from file_path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file {file_path} does not exist.")
            self.file_path = file_path
            self.data = tables.open_file(file_path, mode='r')
        elif isinstance(self.data, tables.File):
            self.file_path = file_path
        else:
            raise ValueError("data must be a path to HDF5 file or a tables.File handle.")

        self.indices = indices
        self.num_classes = num_classes
        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1

        if self.iterations % 10000 == -1 and os.path.exists(self.file_path): # workaround to avoid memory issues
            self.data.close()
            self.data = tables.open_file(self.file_path, mode='r')

        if self.indices is not None:
            if idx < 0 or idx >= len(self.indices):
                raise IndexError(f"Index:{idx} out of bounds for SeqCropsDataset (len(indices): {len(self.indices)}).")
            idx = self.indices[idx]
        
        if idx < 0 or idx >= len(self.data.root.crop_feats):
            raise IndexError(f"Index:{idx} out of bounds for SeqCropsDataset (len(data): {len(self.data.root.crop_feats)}).")

        crop_feat = self.data.root.crop_feats[idx]
        sample_feat = {
            'seq': crop_feat['seq'], # .decode(),
            'seq_onehot': crop_feat['seq_onehot'],
            'cls': crop_feat['cls'] if self.num_classes is None else np.minimum(crop_feat['cls'], self.num_classes - 1),
            'psi': crop_feat['psi'], # usage
            'chrom': crop_feat['chrom'].decode(),
            'start': crop_feat['crop_start'][0],
            'end': crop_feat['crop_end'][0],
            'strand': crop_feat['strand'].decode(),
        }

        sample_feat['cls'] = sample_feat['cls'].astype(np.int64)

        return sample_feat

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.data.root.crop_feats)
    
    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in SeqCropsDataset.")


class GeneSitesDataset(Dataset):
    def __init__(self, meta_df, epoch_size=None, summarize=False,
                 data_dir='./', enable_cache=False, cache_dir='/tmp',
                 weighted_sampling=False, dynamic_weights=True, min_quality=None,
                 stratified_sampling=None, sampling_method=None, 
                 random_seed=42, stage=False, samples_per_seq=1,
                 **kwargs):
        """
        PyTorch Dataset for RNA splice sites at the gene level.

        Args:
            meta_df (pd.DataFrame or str): Metadata DataFrame or path to CSV/pickle file.
            epoch_size (int, optional): Number of samples per epoch. Defaults to length of meta_df.
            data_home (str): Directory containing RNA structure files.
            cache_home (str): Directory for caching data.
            weighted_sampling (bool): Use weighted sampling based on 'sampling_weight' column.
            dynamic_weights (bool): Update weights dynamically during training.
            min_quality (float): Minimum quality threshold for samples.
            stratified_sampling (str, optional): Column name for stratified sampling. If None, no stratification.
            sampling_method (str, optional): Sampling method ('random' or 'sequential'). Not implemented.
            seed_start (int): Starting seed for random sampling.
            training (bool): Whether the dataset is used for training.
            samples_per_seq (int): Number of samples to generate per sequence.
            **kwargs: Additional keyword arguments.
        """
        super(GeneSitesDataset, self).__init__()

        self.is_child_process = utils.get_rank() # utils.is_child_process()

        if isinstance(meta_df, str):
            if not os.path.exists(meta_df):
                raise FileNotFoundError(f"Metadata file {meta_df} does not exist.")
            meta_df = pd.read_csv(meta_df) if meta_df.endswith('csv') else pd.read_pickle(meta_df)
        elif not isinstance(meta_df, pd.DataFrame):
            raise ValueError("meta_df must be a pandas DataFrame.")
        
        self.meta_df = meta_df.reset_index(drop=True)
        self.epoch_size = len(meta_df) if epoch_size is None else epoch_size
        self.random_seed = random_seed
        self.iterations = 0

        self.data_dir = data_dir
        if self.data_dir and not os.path.exists(data_dir):
            raise FileNotFoundError(f"Home directory {data_dir} does not exist.")

        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache_feats = {}
            self.cache_dir = cache_dir
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
       
        self.stage = stage
        self.min_quality = min_quality

        # Set up customized sampling options

        if stratified_sampling:
            ilogger.info(f"Setup stratified sampling based on column: {stratified_sampling}")
            meta_groups = self.meta_df.groupby(stratified_sampling)
            if meta_groups.ngroups <= 1:
                raise ValueError(f"Only one group found for stratified_sampling: {stratified_sampling}. Check your meta_df.")

            self.stratified_sampling = True
            group_indices = meta_groups.indices
            self.stratified_ngroups = len(group_indices)
            self.stratified_names = list(group_indices.keys())
            self.stratified_ilocs = list(group_indices.values())
            ilogger.info(f"Found {self.stratified_ngroups} groups for stratified sampling, e.g., {self.stratified_names[:5]}...")
        else:
            self.stratified_sampling = False
            self.all_ilocs = np.arange(len(meta_df))

        self.weighted_sampling = weighted_sampling
        if self.weighted_sampling:
            if self.weighted_sampling not in meta_df.columns:
                raise ValueError("weighted_sampling is True but column <{}> is missing in meta_df.".format(self.weighted_sampling))
            self.iweight = self.meta_df.columns.get_loc(self.weighted_sampling)
            self.get_normalized_sampling_weights()
        else:
            self.iweight = None  # Default to None if not present

        self.dynamic_weights = dynamic_weights

        self.sampling_method = sampling_method # not yet implemented
        if sampling_method:
            if sampling_method not in ['random', 'sequential']:
                raise ValueError("sampling_method must be 'random' or 'sequential'.")

        self.customized_sampling = (self.stratified_sampling or
                                    self.sampling_method or 
                                    self.weighted_sampling)

        if not self.customized_sampling and self.epoch_size > len(meta_df):
            ilogger.warning(f"epoch_size {self.epoch_size} is greater than sample_size {len(meta_df)}, adjusting epoch_size.")
            self.epoch_size = len(meta_df)

        if summarize and not self.is_child_process:
            # display the state of the dataset in a nice format
            print(f"SpliceDataset summary:")
            print(f"               training: {self.stage}")
            print(f"           len(meta_df): {len(self.meta_df)}")
            print(f"             epoch_size: {self.epoch_size}")
            print(f"               data_dir: {self.data_dir}")
            print(f"              cache_dir: {self.cache_dir if self.enable_cache else 'None'}")
            print(f"            min_quality: {self.min_quality}")
            print(f"    customized_sampling: {self.customized_sampling}")
            print(f"    stratified_sampling: {self.stratified_sampling}")
            print(f"     stratified_ngroups: {self.stratified_ngroups}, e.g., {self.stratified_names[:5]}...")
            print(f"      weighted_sampling: {self.weighted_sampling}")
            print(f"        dynamic_weights: {self.dynamic_weights}")
            # print(f"    sampling_method: {self.sampling_method if self.sampling_method else 'None'}")
            # print(f"    seed_start: {seed_start}")
            # print(f"  samples_per_seq: {samples_per_seq}")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.epoch_size:
            raise IndexError(f"Index:{idx} out of bounds for SpliceDataset (epoch_size: {self.epoch_size}).")

        sample_feat = None
        while sample_feat is None:
            sample_feat = self.omni_sampler(idx)
            
        return sample_feat

    def get_normalized_sampling_weights(self):
        """ Get normalized sampling weights for stratified or all indices """
        if self.stratified_sampling:
            self.stratified_sampling_weights = []
            for i, ilocs in enumerate(self.stratified_ilocs):
                weights = self.meta_df.iloc[ilocs, self.iweight].values
                total_weight = weights.sum()
                if total_weight == 0:
                    ilogger.warning(f"Total sampling weight is zero for stratified group: {self.stratified_names[i]}!")
                    self.stratified_sampling_weights.append(None)
                else:
                    self.stratified_sampling_weights.append(weights / total_weight)
        else:
            weights = self.meta_df.iloc[self.all_ilocs, self.iweight].values
            total_weight = weights.sum()
            if total_weight == 0:
                ilogger.warning(f"Total sampling weight is zero for the entire dataset.")
                self.sampling_weights = None
            else:
                self.sampling_weights = weights / total_weight

    def omni_sampler(self, idx):
        """ Customized sampling logic based on the configuration """
        self.iterations += 1

        if idx >= len(self.meta_df):
            idx = idx % len(self.meta_df)

        if self.customized_sampling:
            if self.stratified_sampling: # randomly sample from stratified groups
                igroup = np.random.randint(self.stratified_ngroups)
                ilocs = self.stratified_ilocs[igroup]
            else:
                ilocs = self.all_ilocs

            if self.weighted_sampling:
                if self.stratified_sampling:
                    weights = self.stratified_sampling_weights[igroup]
                else:
                    weights = self.sampling_weights
                idx = np.random.choice(ilocs, p=weights)
            else:
                idx = np.random.choice(ilocs)

        # check cache_feats if idx exists
        if self.enable_cache and idx in self.cache_feats and len(self.cache_feats[idx]):
            sample_feat = self.cache_feats[idx].pop(-1)
        else:
            gene_ds = self.meta_df.iloc[idx]
            gene_feat = splice_feats.sprinkle_sites_on_sequence(
                gene_ds, 
                # training=self.training,
                )
            crop_feats = splice_feats.get_crop_feats_single_seq(
                gene_feat,
                num_crops=11 if self.enable_cache else 1,
                crop_size=5000,
                flank_size=5000,
                min_sites=0,
                min_usage=0,
                )
            if self.enable_cache and len(crop_feats) > 1:
                self.cache_feats[idx] = crop_feats
                sample_feat = self.cache_feats[idx].pop(-1)
            else:
                sample_feat = crop_feats[0]
        
        if sample_feat is None:
            if self.weighted_sampling and self.dynamic_weights:
                self.meta_df.iloc[idx, self.iweight] = 0.0
            ilogger.warning(f"Warning: Sample features for index {idx} are None, skipping.")
            return None

        if self.dynamic_weights:
            # self.meta_df.iloc[idx, self.iweight] = sample_feats['target_feats']['quality'].item()
            if self.weighted_sampling and self.iterations % self.epoch_size == 0:
                self.get_normalized_sampling_weights()

        if self.stage:
            pass
            # if sample_feats['target_feats']['quality'] < min_quality:
            #     return None
            # if sample_feats['input_feats']['ss'].sum() < 7:
            #     return None
                
        return sample_feat
    
