import os
import time
import datetime
from beartype.typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import wandb

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import TQDMProgressBar

from . import utils
ilogger = utils.get_pylogger(__name__)

def summarize_tensors(vars_dict, prefix=''):
    """ Display variables (a dict of tensors) in a formatted table. """
    if prefix:
        print(f"\n{prefix} Variables:")
    else:
        print("\nVariables:")

    print(f"{'Key':<20} {'Type':<20} {'Shape':<25} {'Dtype':<15} {'Device':<15} {'Requires Grad':<15}")
    print("-" * 110)
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            shape = str(tuple(v.shape))
            dtype = str(v.dtype)
            device = str(v.device)
            vtype = "Tensor"
            requires_grad = str(v.requires_grad)
        else:
            shape = "-"
            dtype = "-"
            device = "-"
            vtype = type(v).__name__
            requires_grad = "-"
        print(f"{k:<20} {vtype:<20} {shape:<25} {dtype:<15} {device:<15} {requires_grad:<15}")
    print("-" * 110)


def summarize_epoch_metrics(epoch_metrics, prefix='', logger=None, logger_prefix='val/epoch_'):
    """ Display epoch metrics (a list of dicts) in a formatted table. """

    if not epoch_metrics or len(epoch_metrics) == 0:
        print("No epoch metrics to summarize.")
        return

    print(f"\n{prefix} Epoch Metrics:")

    avg_metrics = epoch_metrics[0]
    for batch_metrics in epoch_metrics[1:]:
        for k, v in batch_metrics.items():
            if k in avg_metrics:
                avg_metrics[k] += v
            else:
                avg_metrics[k] = v
    avg_metrics = {k: v / len(epoch_metrics) for k, v in avg_metrics.items()}

    for k, v in avg_metrics.items():
        print(f"{k:>28}: {float(v):>9.4f}")

    if logger is not None:
        for metric_name, metric_val in avg_metrics.items():
            logger(f'{logger_prefix}{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
            )


class NanGradientCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Called right before an optimizer step to check gradients for NaNs."""
        if self._any_nan_gradients(pl_module):
            trainer.logger.log_metrics({"nan_gradient_detected": 1}, step=trainer.global_step)
            pl_module.zero_grad(set_to_none=True)  # Clear gradients
            print(f"NaN gradients detected at global step {trainer.global_step}, skipping optimizer step")

    def _any_nan_gradients(self, pl_module):
        """Check if any gradient is NaN in the model"""
        for param in pl_module.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
        return False    
    
            
class OmniDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset=None, test_dataset=None, predict_dataset=None,
                 batch_size=4, num_workers=4, sampler=None, collater=None,
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

        self._extra_cfg = kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self._extra_cfg.get('train_batch_size', self.batch_size),
                          num_workers=self._extra_cfg.get('train_num_workers', self.num_workers),
                          shuffle=self._extra_cfg.get('train_shuffle', True),
                          pin_memory=self._extra_cfg.get('train_pin_memory', True),
                          )
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self._extra_cfg.get('val_batch_size', self.batch_size),
                          num_workers=self._extra_cfg.get('val_num_workers', self.num_workers),
                          shuffle=self._extra_cfg.get('val_shuffle', False),
                          pin_memory=self._extra_cfg.get('val_pin_memory', True),
                          )
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self._extra_cfg.get('test_batch_size', self.batch_size),
                          num_workers=self._extra_cfg.get('test_num_workers', self.num_workers),
                          shuffle=self._extra_cfg.get('test_shuffle', False),
                          pin_memory=self._extra_cfg.get('test_pin_memory', False),
                          )
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          batch_size=self._extra_cfg.get('predict_batch_size', self.batch_size),
                          num_workers=self._extra_cfg.get('predict_num_workers', self.num_workers),
                          shuffle=self._extra_cfg.get('predict_shuffle', False),
                          pin_memory=self._extra_cfg.get('predict_pin_memory', False),
                          )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class OmniMainModule(LightningModule):
    def __init__(self, model, cfg=None, **kwargs):
        super().__init__()

        self.model = model
        self.cfg = {} if cfg is None else cfg
        self.extra_cfg = kwargs

        self.train_epoch_metrics = []
        self.train_epoch_samples = []
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []        
        
        self._train_start_time = time.time()
        self._epoch_start_time = time.time()        

        self._train_steps = 0
        self._validation_steps = 0
        self._predict_steps = 0

        self.save_hyperparameters(ignore=['model'])

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def forward(self, batch_feats):
        return self.model(batch_feats)
    
    def configure_optimizers(self):

        optimizer_cfg = self.cfg.get('optimizer', {})

        lr_scheduler = optimizer_cfg.get('lr_scheduler', None)
        learning_rate = optimizer_cfg.get('learning_rate', 3e-4)
        eps = optimizer_cfg.get('eps', 1e-8)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, eps=eps)

        #  if self.last_lr_step != -1:
        #     for group in optimizer.param_groups:
        #         if 'initial_lr' not in group:
        #             group['initial_lr'] = learning_rate   

        if lr_scheduler is None:
            return optimizer
        
        if lr_scheduler.upper() == "AlphaFoldLRScheduler".upper():
            lr_scheduler = "AlphaFoldLRScheduler"
            # LRScheduler = AlphaFoldLRScheduler(
            #     optimizer,
            #     base_lr=0.0,
            #     max_lr=learning_rate,
            #     last_epoch=self.last_lr_step,
            #     warmup_no_steps=self.args.train_epoch_len * self.args.lr_warmup_end,
            #     start_decay_after_n_steps=self.args.train_epoch_len * self.args.lr_decay_start,
            #     decay_every_n_steps=self.args.train_epoch_len * self.args.lr_decay_inteval,
            #     decay_factor=self.args.lr_decay_factor,
            # )
        elif lr_scheduler.upper() == "CosineAnnealingWarmRestarts".upper():
            LRScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=optimizer_cfg.get('lr_T_0', 3), # in epochs
                T_mult=optimizer_cfg.get('lr_T_mult', 2), # T_i+1 = T_i * T_mult
                eta_min=0.0, # minimum learning rate
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "StepLR".upper():
            LRScheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=optimizer_cfg.get('lr_step_size', 10), # in epochs
                gamma=optimizer_cfg.get('lr_gamma', 0.1), # decay rate
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "MultiStepLR".upper():
            LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optimizer_cfg.get('lr_milestones', [10, 20]),
                gamma=optimizer_cfg.get('lr_gamma', 0.1),
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "CosineAnnealingLR".upper():
            LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=optimizer_cfg.get('lr_T_max', 50), # in epochs
                eta_min=optimizer_cfg.get('lr_eta_min', 0),
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "ReduceLROnPlateau".upper():
            LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=optimizer_cfg.get('lr_gamma', 0.1),
                patience=optimizer_cfg.get('lr_patience', 10),
                threshold=optimizer_cfg.get('lr_threshold', 1e-4),
                threshold_mode='rel',
                cooldown=optimizer_cfg.get('lr_cooldown', 0),
                min_lr=optimizer_cfg.get('lr_min', 0),
                eps=optimizer_cfg.get('lr_eps', 1e-8),
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

        print(f'Using {lr_scheduler} learning rate scheduler...')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LRScheduler,
                "interval": optimizer_cfg.get('lr_interval', 'epoch'), # 'step' or 'epoch'
                "name": f'trainer/{lr_scheduler}',
            }
        }

    def on_train_epoch_start(self):
        ## if <1% 
        # if not self.use_f1 and self.current_epoch >= self.switch_epoch:
            # self.use_f1 = True
            # print(f"🔁 Switching to soft F1 loss at epoch {self.current_epoch}")
        return super().on_train_epoch_start()

    def training_step(self, batch_feats, batch_idx=None):
        if self._train_steps == 0:
            summarize_tensors(batch_feats, prefix='Training Step Input')

        preds = self.model(batch_feats)
        if self._train_steps == 0:
            summarize_tensors(preds, prefix='Training Predictions')

        loss, loss_items = self.model.calc_loss(preds, batch_feats)
        loss_items.update(self.model.calc_metric(preds, batch_feats))
        if self._train_steps == 0:
            summarize_tensors(loss_items, prefix='Training Loss Items')
        
        self.log("train/loss", loss, prog_bar=True)
        self.train_epoch_metrics.append(loss_items)
        self._train_steps += 1
        return loss
    
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self._epoch_start_time = time.time()
        # self.log("trainer/epoch", self.current_epoch, prog_bar=False, logger=True)

        if len(self.train_epoch_metrics) > 0:
            summarize_epoch_metrics(
                self.train_epoch_metrics,
                prefix='Training',
                logger=self._log_scalar,
                logger_prefix='train/epoch_'
            )
            self.train_epoch_metrics.clear()                

    def on_validation_epoch_start(self):
        return super().on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch_feats, batch_idx=None):
        
        preds = self.model(batch_feats)

        loss, loss_items = self.model.calc_loss(preds, batch_feats)
        loss_items.update(self.model.calc_metric(preds, batch_feats))

        self.log("val/loss", loss, prog_bar=True)
        self.validation_epoch_metrics.append(loss_items)
        self._validation_steps += 1
        return loss
        
    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'val/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self._epoch_start_time = time.time()

        if len(self.validation_epoch_metrics) > 0:
            summarize_epoch_metrics(
                self.validation_epoch_metrics,
                prefix='Validation',
                logger=self._log_scalar,
                logger_prefix='val/epoch_'
            )
            self.validation_epoch_metrics.clear()

        return super().on_validation_epoch_end()

    @torch.no_grad()
    def predict_step(self, batch_feats, batch_idx=None):
        if self._predict_steps == 0:
            summarize_tensors(batch_feats, prefix='Predict Step Input')

        preds = self.model.predict(batch_feats)
        if self._predict_steps == 0:
            summarize_tensors(preds, prefix='Predict Step Output')    

        self._predict_steps += 1
        return (batch_feats, preds)


    def fit(self, datamodule, cfg=None, devices=None, run_cfg=None, debug=False, **kwargs):
        """ Fit the model using PyTorch Lightning Trainer.
        Args:
            datamodule: LightningDataModule
            cfg: DictConfig, configuration for training, if None, use self.cfg
            devices: list of int, GPU device ids to use, if None, use all available
            run_cfg: DictConfig, configuration for the entire run, soley for the purpose of saving
        """

        if devices is None:
            devices = [0]

        cfg = self.cfg if cfg is None else \
            OmegaConf.merge(self.cfg, cfg, OmegaConf.create(kwargs))
            
        if cfg.get('warm_start', None) is not None and cfg.get('warm_start_cfg_override', False):
            ilogger.info(f"Warm starting from {cfg.warm_start} with config override...")
            pass # to be implemented
        elif cfg.get('warm_start', None) is not None:
            ilogger.info(f"Warm starting from {cfg.warm_start} without config override...")

        torch_model = self.model
        checkpoints_cfg = cfg['checkpoints']
        date_string =  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        callbacks = []
        ckpt_home = checkpoints_cfg.get('dirpath', 'checkpoints') # keep the original ckpt home

        if debug:
            ilogger.info("Debug mode (without wandb and callbacks)...")
            wandb_logger = None
            save_dir = os.path.join(ckpt_home, f'debug_{date_string}')
            checkpoints_cfg['dirpath'] = save_dir
        else:
            cfg.wandb.name = f'{torch_model.__class__.__name__}_{date_string}'
            wandb_logger = WandbLogger(**cfg.wandb)
            ilogger.info(f"Wandb logger initialized with name: {wandb_logger.experiment.name}, id: {wandb_logger.experiment.id}")

            if cfg.warm_start is None:
                save_dir = os.path.join(ckpt_home, f'{cfg.wandb.name}_{wandb_logger.experiment.id}')
            else:
                save_dir = os.path.dirname(cfg.warm_start)
                # ckpt_dir = os.path.join(ckpt_dir, wandb_logger.experiment.id, cfg.run.warm_start)

            # Model checkpoints
            checkpoints_cfg['dirpath'] = save_dir
            ilogger.info(f"Checkpoints saved to {save_dir}")
            callbacks.append(ModelCheckpoint(**checkpoints_cfg))
            callbacks.append(LearningRateMonitor(logging_interval='step'))
            callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))
            callbacks.append(NanGradientCallback())
            # if cfg.run.trainer.progress_bar_refresh_rate > 0:
                # callbacks.append(RichProgressBar(refresh_rate=cfg.run.trainer.progress_bar_refresh_rate))
                # callbacks.append(TQDMProgressBar(refresh_rate=cfg.run.trainer.progress_bar_refresh_rate))
            # if cfg.run.trainer.detect_anomaly:
                # torch.autograd.set_detect_anomaly(True)
                # ilogger.warning("Anomaly detection is enabled. Training will be slow!")
                #

        # run_conf is solely for saving purposes
        if run_cfg is not None:
            save_cfg = OmegaConf.merge(run_cfg, OmegaConf.create({'litrun': cfg}))
        else:
            save_cfg = cfg

        # Save config to wandb and as a yaml file
        if wandb_logger is not None and isinstance(wandb_logger.experiment.config, wandb.sdk.wandb_config.Config):
            cfg_dict = OmegaConf.to_container(save_cfg, resolve=True)
            flat_cfg = dict(utils.flatten_dict(cfg_dict))
            wandb_logger.experiment.config.update(flat_cfg)

        os.makedirs(save_dir, exist_ok=True)
        cfg_path = os.path.join(save_dir, 'train.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=save_cfg, f=f.name)

        checkpoints_cfg['dirpath'] = ckpt_home # restore the orignal home

        try:
            get_ipython
            cfg.trainer.strategy = 'auto'
            cfg.trainer.deterministic = False
            # cfg.run.trainer.fast_dev_run = True
        except NameError:
            pass

        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            logger=wandb_logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )

        trainer.fit(
            model=self,
            datamodule=datamodule,
            ckpt_path=cfg.get('warm_start', None),
        )

        return trainer