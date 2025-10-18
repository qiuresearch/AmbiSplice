import os
import tables
import datetime
import numpy as np
import pandas as pd
import hydra
import GPUtil
import omegaconf

import torch
import torch.nn as nn

from AmbiSplice import utils
from AmbiSplice import models
from AmbiSplice import datasets
from AmbiSplice import litmodule
from AmbiSplice import loss_metrics
from AmbiSplice import tensor_utils

ilogger = utils.get_pylogger(__name__)

def get_accelerator_devices(gpus=[0]):
    """ Determine the devices to be used for training."""

    if not isinstance(gpus, (list, tuple, omegaconf.listconfig.ListConfig)):
        gpus = [] if gpus is None else [gpus]

    if gpus and torch.cuda.is_available():
        accelerator = str(torch.device("cuda"))
        devices = GPUtil.getAvailable(order='memory', limit=1000)
        devices = [gpu for gpu in gpus if gpu in devices]
        if not devices:
            raise ValueError(f"None of the requested GPUs {gpus} are available!")
        # torch.backends.cudnn.benchmark = True
    else:
        print("GPUs not requested or CUDA is not available, using CPU!")
        accelerator = str(torch.device("cpu"))
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

    if model_cfg.type.upper() == 'SpliceSingle'.upper():
        torch_model = models.SpliceSingle(L=L, W=W, AR=AR, CL=CL, **model_cfg)
    elif model_cfg.type.upper() == 'Pangolin'.upper():
        torch_model = models.Pangolin(L=L, W=W, AR=AR, **model_cfg)
    elif model_cfg.type.upper() == 'PangolinSingle'.upper():
        torch_model = models.PangolinSingle(L=L, W=W, AR=AR, **model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")

    if model_cfg.state_dict_path is not None:
        state_dict_path = os.path.abspath(os.path.expanduser(model_cfg.state_dict_path))
        if not os.path.isfile(state_dict_path):
            raise ValueError(f"Model state_dict path {state_dict_path} does not exist!")
        ilogger.info(f"Loading model state_dict from {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        torch_model.load_state_dict(state_dict, strict=False)
    return torch_model


def get_datasets(dataset_cfg: omegaconf.DictConfig):
    """ Prepare training, validation, and prediction datasets."""

    train_set, val_set, test_set, predict_set = None, None, None, None
    assert dataset_cfg.test_path is None, "test_path is not used in this script, please use predict_path instead!"

    if dataset_cfg.type.upper() == 'GeneSites'.upper():
        meta_df_path = dataset_cfg.train_path
        ilogger.info(f"Loading metadata from {meta_df_path}")
        meta_df = pd.read_pickle(meta_df_path)

        test_chroms = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        train_chroms = ['chr2', 'chr4', 'chr6', 'chr8', 'chr10',
                        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

        isin_test = meta_df['chrom'].isin(test_chroms)
        test_df = meta_df[isin_test].reset_index(drop=True)
        train_val_df = meta_df[~ isin_test].reset_index(drop=True)

        # A poor man's method to split train and val using chrom column for stratification
        chrom_groups = train_val_df.groupby('chrom')
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

        train_set = datasets.GeneSitesDataset(train_df, epoch_size=dataset_cfg.train_size, summarize=True, **dataset_cfg)
        val_set = datasets.GeneSitesDataset(val_df, epoch_size=dataset_cfg.val_size, summarize=False, **dataset_cfg)
        # test_set = datasets.GeneSitesDataset(test_df, epoch_size=dataset_cfg.test_size, summarize=False, **dataset_cfg)
        predict_set = datasets.GeneSitesDataset(test_df, epoch_size=dataset_cfg.predict_size, summarize=False, **dataset_cfg)
    elif dataset_cfg.type.upper() == 'GeneCrops'.upper():
        train_indices, val_indices = None, None

        if dataset_cfg.train_path:
            train_set = datasets.GeneCropsDataset(dataset_cfg.train_path, file_path=dataset_cfg.train_path, epoch_size=dataset_cfg.train_size, summarize=True, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith('train'):
            raise ValueError("Training stage requires train_path to be specified in dataset config!")

        if dataset_cfg.val_path:
            val_set = datasets.GeneCropsDataset(dataset_cfg.val_path, file_path=None, epoch_size=dataset_cfg.val_size, summarize=False, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith('train') and train_set is not None:
            # Create validation set from training set
            # The easiest is to use torch.utils.data.random_split. 
            # Here we can think about doing stratified splitting, etc. later.
            all_indices = np.arange(len(train_set))
            train_split_size = int(0.9 * len(train_set))
            np.random.seed(dataset_cfg.split_seed if 'split_seed' in dataset_cfg else 111)
            np.random.shuffle(all_indices)
            train_indices = all_indices[:train_split_size]
            val_indices = all_indices[train_split_size:]
            ilogger.info(f"Split training set into train ({len(train_indices)}) and val ({len(val_indices)}) subsets.")

            val_set = torch.utils.data.Subset(train_set, val_indices)
            train_set = torch.utils.data.Subset(train_set, train_indices)
        elif dataset_cfg.stage.lower().startswith('train'):
            ilogger.warning("No validation dataset used for training!!!")
            
        if dataset_cfg.predict_path:
            predict_set = datasets.GeneCropsDataset(dataset_cfg.predict_path, file_path=None, epoch_size=dataset_cfg.predict_size, summarize=False, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith(('eval', 'predict')):
            raise ValueError(f"{dataset_cfg.stage} stage requires predict_path to be specified in dataset config!")

    elif dataset_cfg.type.upper() == 'Pangolin'.upper():

        if dataset_cfg.train_path:
            train_set = datasets.PangolinDataset(file_path=dataset_cfg.train_path, epoch_size=dataset_cfg.train_size, summarize=True, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith('train'):
            raise ValueError("Training stage requires train_path to be specified in dataset config!")

        if dataset_cfg.val_path:
            val_set = datasets.PangolinDataset(file_path=dataset_cfg.val_path, epoch_size=dataset_cfg.val_size, summarize=False, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith('train') and train_set is not None:
            train_split_size = int(0.9 * len(train_set))
            train_set, val_set = torch.utils.data.random_split(train_set, [train_split_size, len(train_set) - train_split_size],
                        generator=torch.Generator().manual_seed(dataset_cfg.split_seed if 'split_seed' in dataset_cfg else 42))
            ilogger.info(f"Split training set into train ({len(train_set)}) and val ({len(val_set)}) subsets.")
        elif dataset_cfg.stage.lower().startswith('train'):
            ilogger.warning("No validation dataset used for training!!!")

        if dataset_cfg.predict_path:
            predict_set = datasets.PangolinDataset(file_path=dataset_cfg.predict_path, epoch_size=dataset_cfg.predict_size, summarize=False, **dataset_cfg)
        elif dataset_cfg.stage.lower().startswith(('eval', 'predict')):
            raise ValueError(f"{dataset_cfg.stage} stage requires predict_path to be specified in dataset config!")

    else:
        raise ValueError(f"Unknown dataset type: {dataset_cfg.type}")

    return {'train': train_set, 'val': val_set, 'test': test_set, 'predict': predict_set}


def get_litdata(datasets, dataloader_cfg: omegaconf.DictConfig):
    litdata = litmodule.OmniDataModule(train_dataset=datasets['train'],
                                       val_dataset=datasets['val'],
                                       test_dataset=datasets['test'],
                                       predict_dataset=datasets['predict'],
                                       **dataloader_cfg)
    return litdata


def get_litrun(cfg: omegaconf.DictConfig, model: nn.Module):
    litrun = litmodule.OmniRunModule(model=model, cfg=cfg)
    return litrun


def get_ensemble_litruns(main_cfg, model=None):
    """ Get ensemble litruns from the main config
        Only handle two config cases: main_cfg.ensemble.model and main_cfgensemble.litrun
    """

    def get_ensemble_cfgs(ensemble_cfg):
        """ Convert ensemble_cfg.model to a list of model cfgs."""
        ilogger.info(f"Ensemble config:\n{omegaconf.OmegaConf.to_yaml(ensemble_cfg)}")
        cfgs = []
        for key, val in ensemble_cfg.items():
            if val is None or val[0] is None:
                continue
            for i, v in enumerate(val):
                if i >= len(cfgs):
                    cfgs.append({})
                cfgs[i][key] = v
        return cfgs

    model_cfgs = get_ensemble_cfgs(main_cfg.ensemble.model) if main_cfg.ensemble.model is not None else None
    litrun_cfgs = get_ensemble_cfgs(main_cfg.ensemble.litrun) if main_cfg.ensemble.litrun is not None else None

    ensemble_size = max(len(model_cfgs) if model_cfgs else 0,
                        len(litrun_cfgs) if litrun_cfgs else 0)
    if ensemble_size == 0:
        raise ValueError("Ensemble size is zero! Please provide ensemble.model and/or ensemble.litrun.")

    ilogger.info(f"Ensemble size: {ensemble_size}")
    # get the list of models (which may be the same model)
    if model_cfgs:
        assert len(model_cfgs) == ensemble_size, "Length of ensemble.model does not match ensemble size!"
        models = [get_torch_model(omegaconf.OmegaConf.merge(main_cfg.model, omegaconf.OmegaConf.create(cfg))) for cfg in model_cfgs]
    else:
        models = [model for _ in range(ensemble_size)]
    # get the list of litrun (which may be the same litrun)
    if litrun_cfgs:
        assert len(litrun_cfgs) == ensemble_size, "Length of ensemble.litrun does not match ensemble size!"
        litruns = [get_litrun(omegaconf.OmegaConf.merge(main_cfg.litrun, omegaconf.OmegaConf.create(cfg)), model=models[i]) for i, cfg in enumerate(litrun_cfgs)]
    else:
        litruns = [get_litrun(main_cfg.litrun, model=models[i]) for i in range(ensemble_size)]
        
    return litruns
    

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
    #           epoch train_loss val_loss
    #           0 0.00658477892685834 0.004716315679716134
    #           1 0.0045675282197297105 0.004415887669691599
    #           2 0.004811793682136806 0.004618272451537466
    #           3 0.0045143571978752 0.004389756265428374
    #           4 0.0042724353038384255 0.004255377201714269
    #           5 0.004095182792971297 0.004187629859801202
    #           6 0.004493844392550496 0.0045338927623110835
    #           7 0.00440326978519716 0.0044299452549822005
    #           8 0.004298632183430625 0.004294230581608949
    #           9 0.00419190875022965 0.004320022200705696
    #           10 0.004074970081213211 0.004217715382484738
    # 8) eight models are trained separately for four tissues. One for 'cls' and one for 'psi' in each tissue.
    # 9) three models are averaged for final prediction

    # cfg_path = os.path.join(os.getcwd(), 'configs', 'train.yaml')
    # main_cfg = omegaconf.OmegaConf.load(cfg_path)

    omegaconf.OmegaConf.set_struct(main_cfg, False)  # allow new attribute assignment
    accelerator, devices = get_accelerator_devices(gpus=main_cfg.gpus)

    torch_model = get_torch_model(main_cfg.model)
    litrun = get_litrun(main_cfg.litrun, torch_model)

    datasets = get_datasets(main_cfg.dataset)
    litdata = get_litdata(datasets, main_cfg.dataloader)

    if main_cfg.stage in ('train', 'training', 'fit'):
        ilogger.info(f"Training model with config:\n{omegaconf.OmegaConf.to_yaml(main_cfg)}")
        trainer = litrun.fit(litdata, save_cfg=main_cfg, accelerator=accelerator, devices=devices, debug=main_cfg.debug)
        ilogger.info("Training completed.")
        
        return trainer
    elif main_cfg.stage in ('test', 'testing', 'eval', 'evaluate'):

        if not main_cfg.ensemble.enable:
            eval_outputs = litrun.evaluate(datamodule=litdata,
                                           save_prefix=main_cfg.save_prefix, save_individual=main_cfg.save_individual,
                                           accelerator=accelerator, devices=devices, debug=main_cfg.debug)            
        else:
            litruns = get_ensemble_litruns(main_cfg, model=torch_model)
            epoch_feats = None
            ensemble_cls = []
            ensemble_psi = []
            for i, litrun in enumerate(litruns):
                ilogger.info(f"Evaluating ensemble model {i+1}/{len(litruns)}")
                eval_outputs = litrun.evaluate(datamodule=litdata, 
                                               save_prefix=f"{main_cfg.save_prefix}_ens{i+1}", save_individual=False, 
                                               accelerator=accelerator, devices=devices, debug=main_cfg.debug)

                if epoch_feats is None:
                    epoch_feats = eval_outputs[0]  # same for all ensemble members
                ensemble_cls.append(eval_outputs[1]['cls'])  # list of tensors
                ensemble_psi.append(eval_outputs[1]['psi'])  # list of tensors
                
                # delete litrun to save memory
                del litrun
                del eval_outputs
                torch.cuda.empty_cache()

            # average the ensemble outputs
            avg_cls = torch.mean(torch.stack(ensemble_cls, dim=0), dim=0)
            avg_psi = torch.mean(torch.stack(ensemble_psi, dim=0), dim=0)
            eval_outputs = (epoch_feats, {'cls': avg_cls, 'psi': avg_psi})

            save_path = f"{main_cfg.save_prefix}_ens_outputs.pt"
            ilogger.info(f"Saving ensemble outputs to {save_path} ...")
            torch.save(eval_outputs, save_path)
            ilogger.info(f"Ensemble outputs saved to {save_path}")
            
            ensemble_metrics = loss_metrics.calc_benchmark(eval_outputs[1], eval_outputs[0], keep_batchdim=False)
            if main_cfg.save_prefix is not None:
                metrics_path = f"{main_cfg.save_prefix}_ens_metrics.yaml"
                tensor_utils.to_yaml(ensemble_metrics, metrics_path)
                ilogger.info(f"Ensemble summary metrics saved to {metrics_path}")

                if main_cfg.save_individual:
                    ind_metrics = loss_metrics.calc_benchmark(eval_outputs[1], eval_outputs[0], keep_batchdim=True)
                    metrics_path = f"{main_cfg.save_prefix}_ens_ind_metrics.csv"
                    ilogger.info(f"Saving ensemble individual metrics to {metrics_path} ...")
                    loss_metrics.save_individual_metrics(ind_metrics, metrics_path)
                    ilogger.info(f"Ensemble individual metrics saved to {metrics_path}")
            else:
                ilogger.info(f"Ensemble summary metrics:\n{omegaconf.OmegaConf.to_yaml(ensemble_metrics)}")

        ilogger.info("Testing completed.")

        return eval_outputs
    elif main_cfg.stage in ('predict', 'inference', 'pred'):
        pred_outputs = litrun.predict(datamodule=litdata,
                                      save_prefix=main_cfg.save_prefix, save_individual=main_cfg.save_individual, 
                                      accelerator=accelerator, devices=devices, debug=main_cfg.debug)
        ilogger.info("Prediction completed.")

        return pred_outputs
    else:
        raise ValueError(f"Unknown stage: {main_cfg.stage}")
        

if __name__ == "__main__":
    main()