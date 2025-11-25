import os
import h5py
import tables
import torch
from torch.utils.data import Dataset
from omegaconf.listconfig import ListConfig
import numpy as np
import pandas as pd

from . import splice_feats
from . import utils

ilogger = utils.get_pylogger(__name__)


def get_pangolin_tissue_cols(tissue_types):
    # tissue order: heart, liver, brain, testis
    assert isinstance(tissue_types, (list, tuple, ListConfig)) and len(tissue_types) > 0, \
        f"tissue_types: {tissue_types} must be a list or tuple with at least one element."
    tissue_types = [t.lower() if isinstance(t, str) else t for t in tissue_types]
    # starting column for each tissue in Y
    col_map = {
        'heart': 0,
        'liver': 3,
        'brain': 6,
        'testis': 9,
        1: 0,
        2: 3,
        3: 6,
        4: 9,
    }

    tissue_cols = []
    for t in tissue_types:
        if t in col_map:
            tissue_cols.append(col_map[t])
        else:
            raise ValueError(f"Invalid tissue_type: {t}. Must be one of {list(col_map.keys())}.")
    return tissue_cols


class PangolinSoloDataset(Dataset):
    """ Adapted from Pangolin github repo
    Returns sample_feat containing only one tissue type per call, which is randomly picked 
    based on tissue_types list.
    """
    def __init__(self, file_path, tissue_types=['heart'], tissue_embedding_path=None, epoch_length=None, **kwargs):
        super(PangolinSoloDataset, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        assert len(self.data) % 3 == 0
        self.epoch_length = len(self.data) // 3 * len(tissue_types)
        if epoch_length:
            self.epoch_length = min(epoch_length, self.epoch_length)
        
        self.tissue_cols = get_pangolin_tissue_cols(tissue_types)
        self.tissue_types = [t.lower() for t in tissue_types]

        if tissue_embedding_path:
            ilogger.info(f"Loading tissue embeddings from {tissue_embedding_path}...")
            self.tissue_embeddings = pd.read_csv(tissue_embedding_path, index_col=0)
            self.tissue_embeddings.index = [str(idx).lower() for idx in self.tissue_embeddings.index]

            for t in self.tissue_types:
                if t not in self.tissue_embeddings.index:
                    raise ValueError(f"Tissue type {t} not found in tissue embeddings index.")
        else:
            self.tissue_embeddings = None

        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1
        if self.iterations % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.file_path, 'r', libver='latest')

        if len(self.tissue_cols) > 1:
            idx = idx // len(self.tissue_cols)
            itissue = idx % len(self.tissue_cols)
        else:
            itissue = 0

        tissue_col = self.tissue_cols[itissue]

        X = self.data['X' + str(idx)][:].T # transpose to (4, L)
        Y = self.data['Y' + str(idx)][:].T # transpose to (12, L)
        Z = self.data['Z' + str(idx)][:] # (4,): [chrom, start, end, strand]

        # each tissue: unspliced, spliced, usage
        sample_feat = {
            'seq': splice_feats.decode_onehot(X, dim=0, idx2base=np.array(['A', 'C', 'G', 'T', 'N'])),
            'seq_onehot': X.astype(np.float32), # (4, L)
            'cls': np.argmax(Y[tissue_col:tissue_col+2, :], axis=0), # 0: unspliced, 1: spliced (L)
            'psi': Y[tissue_col+2, :].astype(np.float32), # usage (L)
            'chrom': Z[0].decode(),
            'start': int(Z[1]),
            'end': int(Z[2]),
            'strand': Z[3].decode(),
        }

        if self.tissue_embeddings is not None:
            tissue_type = self.tissue_types[itissue]
            tissue_embedding = self.tissue_embeddings.loc[tissue_type].values.astype(np.float32)

            sample_feat['seq_onehot'] = np.concatenate(
                [sample_feat['seq_onehot'], 
                 np.tile(tissue_embedding[:, np.newaxis], (1, sample_feat['seq_onehot'].shape[1]))],
                axis=0)

        return sample_feat

    def __len__(self):
        return self.epoch_length
    
    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in PangolinSoloDataset.")


class PangolinDataset(Dataset):
    """ Adapted from Pangolin github repo """
    def __init__(self, file_path, tissue_types=['heart', 'liver', 'brain', 'testis'], epoch_length=None, **kwargs):
        super(PangolinDataset, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        assert len(self.data) % 3 == 0
        self.epoch_length = len(self.data) // 3
        if epoch_length:
            self.epoch_length = min(epoch_length, self.epoch_length)

        self.tissue_cols = get_pangolin_tissue_cols(tissue_types)
        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1
        if self.iterations % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.file_path, 'r', libver='latest')

        X = self.data['X' + str(idx)][:].T # transpose to (4, L)
        Y = self.data['Y' + str(idx)][:].T # transpose to (12, L)
        Z = self.data['Z' + str(idx)][:] # (4,): [chrom, start, end, strand]

        # for each tissue: unspliced, spliced, usage
        cls_list, psi_list = [], []
        for tissue_col in self.tissue_cols:
            cls_list.append(np.argmax(Y[tissue_col:tissue_col+2, :], axis=0)) # 0: unspliced, 1: spliced
            psi_list.append(Y[tissue_col+2, :]) # usage
        
        sample_feat = {
            'seq': splice_feats.decode_onehot(X, dim=0, idx2base=np.array(['A', 'C', 'G', 'T', 'N'])),
            'seq_onehot': X.astype(np.float32),
            'cls': np.stack(cls_list, axis=0).astype(np.int64), # (num_tissues, L)
            'psi': np.stack(psi_list, axis=0).astype(np.float32), # (num_tissues, L)
            'chrom': Z[0].decode(),
            'start': int(Z[1]),
            'end': int(Z[2]),
            'strand': Z[3].decode(),
        }

        return sample_feat

    def __len__(self):
        return self.epoch_length
    
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
        # Change dtypes as needed because of low-precision storage in HDF5 if possible
        sample_feat = {
            'seq': crop_feat['seq'], # .decode(), # strings are saved in bytes in HDF5
            'seq_onehot': crop_feat['seq_onehot'].astype(np.float32),
            'cls': crop_feat['cls'].astype(np.int64),
            'psi': crop_feat['psi'], # usage
            'chrom': crop_feat['chrom'].decode(),
            'gene_id': crop_feat['gene_id'].decode(),
            'gene_start': crop_feat['gene_start'][0],
            'gene_end': crop_feat['gene_end'][0],
            'strand': crop_feat['strand'].decode(),
            'start': crop_feat['crop_start'][0],
            'end': crop_feat['crop_end'][0],
        }

        if self.num_classes:
            sample_feat['cls'] = np.minimum(crop_feat['cls'], self.num_classes - 1)

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
    def __init__(self, meta_df, epoch_length=None, summarize=False,
                 data_dir='./', enable_cache=False, cache_dir='/tmp',
                 weighted_sampling=False, dynamic_weights=True, min_quality=None,
                 stratified_sampling=None, sampling_method=None, 
                 random_seed=42, training=True,
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
        self.epoch_length = len(meta_df) if epoch_length is None else epoch_length
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
       
        self.training = training
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

        if not self.customized_sampling and self.epoch_length > len(meta_df):
            ilogger.warning(f"epoch_size {self.epoch_length} is greater than sample_size {len(meta_df)}, adjusting epoch_size.")
            self.epoch_length = len(meta_df)

        if summarize and not self.is_child_process:
            print(f"SpliceDataset summary:")
            print(f"               training: {self.training}")
            print(f"           len(meta_df): {len(self.meta_df)}")
            print(f"             epoch_size: {self.epoch_length}")
            print(f"               data_dir: {self.data_dir}")
            print(f"              cache_dir: {self.cache_dir if self.enable_cache else 'None'}")
            print(f"            min_quality: {self.min_quality}")
            print(f"    customized_sampling: {self.customized_sampling}")
            print(f"    stratified_sampling: {self.stratified_sampling}")
            print(f"     stratified_ngroups: {self.stratified_ngroups}, e.g., {self.stratified_names[:5]}...")
            print(f"      weighted_sampling: {self.weighted_sampling}")
            print(f"        dynamic_weights: {self.dynamic_weights}")
            # print(f"    sampling_method: {self.sampling_method if self.sampling_method else 'None'}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.epoch_length:
            raise IndexError(f"Index:{idx} out of bounds for SpliceDataset (epoch_size: {self.epoch_length}).")

        sample_feat = None
        while sample_feat is None:
            sample_feat = self.omni_sampler(idx)
            
        return sample_feat

    def get_normalized_sampling_weights(self):
        """ Get normalized sampling weights for stratified or all indices 
        
        Yields:
            self.stratified_sampling_weights: list of np.arrays for each stratified group
            self.sampling_weights: np.array for all indices         
        """

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

        # if idx >= len(self.meta_df):
        #     idx = idx % len(self.meta_df)

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
            crop_feats = splice_feats.get_crop_feats_from_single_seq(
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

        if self.training:
            if self.dynamic_weights:
                # Todo: update sampling weights based on sample quality
                if self.weighted_sampling and self.iterations % self.epoch_length == 0:
                    self.get_normalized_sampling_weights()
                
        return sample_feat
    
