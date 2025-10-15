"""
Utility functions for experiments.

Code adapted from
https://github.com/microsoft/protein-frame-flow/blob/main/experiments/utils.py
"""

import os
import logging
import torch
from typing import Optional
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def is_child_process():
    return torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() > 0
    

def get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("GLOBAL_RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK") # NODE_RANK RANK
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened