import os
import datetime
import numpy as np
import pandas as pd

# import hydra
import GPUtil
import hydra
import omegaconf

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from AmbiSplice import data_utils
from AmbiSplice import visual_utils
from AmbiSplice import utils
from AmbiSplice import model
from AmbiSplice import dataset
from AmbiSplice import litmodule

ilogger = utils.get_pylogger(__name__)

def get_accelerator_devices(gpus=[0]):
    """ Determine the devices to be used for training."""

    if not isinstance(gpus, (list, tuple, omegaconf.listconfig.ListConfig)):
        gpus = [] if gpus is None else [gpus]

    if gpus and torch.cuda.is_available():
        accelerator = torch.device("cuda")
        devices = GPUtil.getAvailable(order='memory', limit=1000)
        devices = [gpu for gpu in gpus if gpu in devices]
        if not devices:
            raise ValueError(f"None of the requested GPUs {gpus} are available!")
        # torch.backends.cudnn.benchmark = True
    else:
        print("GPUs not requested or CUDA is not available, using CPU!")
        accelerator = torch.device("cpu")
        devices = "auto"

    torch.set_float32_matmul_precision('medium')
    torch.use_deterministic_algorithms(False)
    print(f"Available accelerator: {accelerator}; devices: {devices}")

    return accelerator, devices


def get_torch_model(model_cfg: omegaconf.DictConfig):
    """ Initialize model from config."""
    L = 32
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    # atrous rate in residual units
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                    10, 10, 10, 10, 25, 25, 25, 25])
    CL = 2 * np.sum(AR * (W - 1))
    torch_model = model.PangolinSingle(L=L, W=W, AR=AR, **model_cfg)
    return torch_model


def get_lit_data(dataset_cfg: omegaconf.DictConfig,
                 dataloader_cfg: omegaconf.DictConfig = None):
    """ Prepare training, validation, and prediction datasets."""
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

    train_dataset = dataset.SpliceDataset(train_df, epoch_size=dataset_cfg.train_size, summarize=True, **dataset_cfg)
    val_dataset = dataset.SpliceDataset(val_df, epoch_size=dataset_cfg.val_size, summarize=False, **dataset_cfg)
    predict_dataset = dataset.SpliceDataset(test_df, epoch_size=dataset_cfg.predict_size, summarize=False, **dataset_cfg)

    lit_data = litmodule.OmniDataModule(train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        predict_dataset=predict_dataset,
                                        **dataloader_cfg)
    return lit_data


def get_lit_run(cfg: omegaconf.DictConfig, model: nn.Module):
    lit_run = litmodule.OmniRunModule(model=model, cfg=cfg)
    return lit_run


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(main_cfg: omegaconf.DictConfig):
    # From Pangolin: 
    # 1) no dropout!!!
    # 2) batch_size=12, 90/10 random split for train/val
    # 3) AdamW optimizer with lr=5e-4
    # 4) CosineAnnealingWarmRestarts with T_0=4 (or 2), T_mult=2, manually stepped with epochs in fractionals (more gradual?)
    # 5) cross-entropy loss for classification (two classes)
    # 6) BCELoss for usage while masking off nucleotides with negative usage (assigned in data processing)
    # 7) trained for 10 epochs, little changes in loss after 2 epoches. Saved model after every epoch. Chose best based on val loss.
    # epoch train_loss val_loss
    # 0 0.00658477892685834 0.004716315679716134
    # 1 0.0045675282197297105 0.004415887669691599
    # 2 0.004811793682136806 0.004618272451537466
    # 3 0.0045143571978752 0.004389756265428374
    # 4 0.0042724353038384255 0.004255377201714269
    # 5 0.004095182792971297 0.004187629859801202
    # 6 0.004493844392550496 0.0045338927623110835
    # 7 0.00440326978519716 0.0044299452549822005
    # 8 0.004298632183430625 0.004294230581608949
    # 9 0.00419190875022965 0.004320022200705696
    # 10 0.004074970081213211 0.004217715382484738

    # cfg_path = os.path.join(os.getcwd(), 'configs', 'train.yaml')
    # main_cfg = omegaconf.OmegaConf.load(cfg_path)
    # omegaconf.OmegaConf.set_struct(main_cfg, False)  # allow new attribute assignment

    omegaconf.OmegaConf.set_struct(main_cfg, False)  # allow new attribute assignment
    accelerator, devices = get_accelerator_devices(gpus=main_cfg.gpus)

    ambisplice_model = get_torch_model(main_cfg.model)
    lit_run = get_lit_run(main_cfg.litrun, ambisplice_model)

    lit_data = get_lit_data(main_cfg.dataset, main_cfg.dataloader)

    trainer = lit_run.fit(lit_data, save_cfg=main_cfg, devices=devices, debug=main_cfg.debug)

    return trainer


if __name__ == "__main__":
    main()