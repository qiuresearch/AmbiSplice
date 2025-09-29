import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# from AmbiSplice.data_utils import *
from AmbiSplice import visual_utils
from AmbiSplice import utils
from AmbiSplice import model
from AmbiSplice import dataset
from AmbiSplice import litmodule

import GPUtil
# import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
import wandb


ilogger = utils.get_pylogger(__name__)
# torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(False)

cfg_path = os.path.join(os.getcwd(), 'configs', 'train.yaml')
cfg = OmegaConf.load(cfg_path)
OmegaConf.set_struct(cfg, False)  # allow new attribute assignment

L = 32
# convolution window size in residual units
W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()