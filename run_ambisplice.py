import os
import datetime
import numpy as np
import pandas as pd

# import hydra
import GPUtil
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import TQDMProgressBar

from AmbiSplice import data_utils
from AmbiSplice import visual_utils
from AmbiSplice import utils
from AmbiSplice import model
from AmbiSplice import dataset
from AmbiSplice import litmodule

ilogger = utils.get_pylogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    devices = GPUtil.getAvailable(order='memory')
    # torch.backends.cudnn.benchmark = True
else:
    print("Caution:: CUDA is not available, using CPU!")
    device = torch.device("cpu")
    devices = [0]

torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(False)
print(f"Available {device} devices: {devices}")

# Datasets
meta_df_path = os.path.join(os.getcwd(), 'rna_sites.pkl')
meta_df = pd.read_pickle(meta_df_path)

test_chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
train_chroms = ['chr2', 'chr4', 'chr6', 'chr8', 'chr10',
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

isin_test = meta_df['chrom'].isin(test_chroms)
test_df = meta_df[isin_test].reset_index(drop=True)
train_val = meta_df[~ isin_test].reset_index(drop=True)

# A poor man's method to split train and val using chrom column for stratification
chrom_groups = train_val.groupby('chrom')
train_list = []
val_list = []
for chrom, group in chrom_groups:
    group = group.sort_values(by='len').reset_index(drop=True)
    val_list.append(group.iloc[::6])
    train_list.append(group.drop(group.index[::6]))
    # group = group.sample(frac=1, random_state=42)  # shuffle the group
    # n_val = max(1, int(len(group) * 0.15))  # at least one sample for validation
    # val_list.append(group.iloc[:n_val])
    # train_list.append(group.iloc[n_val:])

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)

# show the sizes of the datasets
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

cfg_path = os.path.join(os.getcwd(), 'configs', 'train.yaml')
cfg = OmegaConf.load(cfg_path)
OmegaConf.set_struct(cfg, False)  # allow new attribute assignment

train_dataset = dataset.SpliceDataset(train_df, epoch_size=cfg.dataset.train_size, **cfg.dataset)
val_dataset = dataset.SpliceDataset(val_df, epoch_size=cfg.dataset.val_size, **cfg.dataset)
predict_dataset = dataset.SpliceDataset(test_df, epoch_size=100, **cfg.dataset)

L = 32
W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])
CL = 2 * np.sum(AR * (W - 1))
ambisplice_model = model.PangolinSingle(L=L, W=W, AR=AR, **cfg.model)
lit_model = litmodule.OmniMainModule(model=ambisplice_model, optimizer_cfg=cfg.run.optimizer, trainer_cfg=cfg.run.trainer)
lit_data = litmodule.OmniDataModule(train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    predict_dataset=predict_dataset,
                                    **cfg.dataloader)