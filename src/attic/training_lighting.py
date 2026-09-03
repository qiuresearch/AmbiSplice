## usage: python training.py --input 102_training_data_sequece_exp.pt --epochs 30 --model exp --output 102_model_exp.pt
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

# Constants
L_DIM = 32
W = np.asarray([11]*8 + [21]*4 + [41]*4)
AR = np.asarray([1]*4 + [4]*4 + [10]*4 + [25]*4)

# Define ResBlock, Pangolin, PangolinEXP, masked_focal_mse, soft_f1_loss here...
class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out
        
class Pangolin(nn.Module):
    def __init__(self, L, W, AR):
        super(Pangolin, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        out1 = F.softmax(self.conv_last1(skip), dim=1)
        out2 = torch.sigmoid(self.conv_last2(skip))
        return torch.cat([out1, out2], 1)

class SplicingEXP(nn.Module):
    def __init__(self, L, W, AR):
        super(SplicingEXP, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(14, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 1, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        
    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        out1 = F.softmax(self.conv_last1(skip), dim=1)
        out2 = torch.sigmoid(self.conv_last2(skip))
        return torch.cat([out1, out2], 1)

class PangolinEXP(nn.Module):
    def __init__(self, L, W, AR):
        super(PangolinEXP, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(14, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        out1 = self.conv_last1(skip)
        out2 = torch.sigmoid(self.conv_last2(skip))
        return torch.cat([out1, out2], 1)

def sharpened_focal_mse(pred, target, gamma=2.0, min_target=0.01, peak_boost=2.0):
    # Only focus on meaningful (non-zero) target positions
    mask = (target > min_target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Error
    error = (pred[mask] - target[mask]).pow(2)

    # Sharpening weight: boost near high-usage (peak) values
    target_vals = target[mask]
    sharpen_weight = (target_vals ** peak_boost)  # prioritize sharp peaks
    focal_weight = torch.abs(pred[mask] - target_vals).pow(gamma)

    # Combine weights
    weight = sharpen_weight * focal_weight
    return torch.mean(weight * error)

def masked_focal_mse(pred, target, gamma=2.0, min_target=0.01):
    mask = (target > min_target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    error = (pred[mask] - target[mask]).pow(2)
    weight = torch.abs(pred[mask] - target[mask]).pow(gamma)
    return torch.mean(weight * error)

# Soft F1 loss
def soft_f1_loss(y_pred_logits, y_true, epsilon=1e-7):
    """
    y_pred_logits, y_true: [B, 3, T]
    Focus loss on channels 0 and 1 (binary), ignore or reduce weight on channel 2 (regression).
    """
    y_pred_logits = torch.clamp(y_pred_logits, 0, 1)
    
    f1_losses = []
    weights = [1.0, 1.0, 0.1]  # downweight channel 2

    for i in range(3):
        tp = torch.sum(y_true[:, i, :] * y_pred_logits[:, i, :], dim=[1])
        fp = torch.sum((1 - y_true[:, i, :]) * y_pred_logits[:, i, :], dim=[1])
        fn = torch.sum(y_true[:, i, :] * (1 - y_pred_logits[:, i, :]), dim=[1])
        f1 = 2 * tp / (2 * tp + fp + fn + epsilon)
        f1_losses.append(weights[i] * (1 - f1))

    return torch.mean(sum(f1_losses))

def weighted_mse_loss(pred, target, threshold=0.5, low_weight=1.0, high_weight=1000.0):
    """
    Weighted MSE loss that puts higher weight on targets > threshold.
    
    Args:
        pred: predicted tensor [B, T]
        target: ground truth tensor [B, T]
        threshold: float, target value cutoff for higher weighting
        low_weight: weight for target <= threshold
        high_weight: weight for target > threshold
    """
    
    high_mask = (target > threshold).float()
    low_mask = 1.0 - high_mask
    
    weight_mask = low_weight * low_mask + high_weight * high_mask
    loss = weight_mask * (pred - target) ** 2
    return loss.mean()


def hybrid_loss(y_pred_logits, y_true, weight_f1=1.0, weight_bce=1.0, weight_usage=1.0):
    """
    Combines BCE for binary channels [0, 1] and MSE for usage [2].
    Optionally includes soft F1 as a regularizer for [0, 1].
    """
    # Clamp predictions to valid range
    y_pred_logits = torch.clamp(y_pred_logits, 0, 1)

    # Binary cross entropy for channels 0 and 1
    #bce_0 = F.binary_cross_entropy(y_pred_logits[:, 0, :], y_true[:, 0, :])
    #bce_1 = F.binary_cross_entropy(y_pred_logits[:, 1, :], y_true[:, 1, :])
    #bce_loss = bce_0 + bce_1

    # Convert two-channel labels to single class index: (1,0)->0, (0,1)->1
    y_cls = torch.argmax(y_true[:, 0:2, :], dim=1)  # shape: (batch, seq_len)
    logits = y_pred_logits[:, 0:2, :].permute(0, 2, 1)     # shape: (batch, seq_len, 2)
    ce_loss = F.cross_entropy(logits.reshape(-1, 2), y_cls.reshape(-1))
    #F.cross_entropy(y_pred_logits, )
    
    # Mean squared error for usage prediction (channel 2)
    mse_usage = weighted_mse_loss(y_pred_logits[:, 2, :], y_true[:, 2, :])
    #F.mse_loss(y_pred_logits[:, 2, :], y_true[:, 2, :])

    # Optional: Soft F1 for channels 0 and 1 (as a regularizer)
    f1_loss = 0
    y_pred_logits = F.softmax(y_pred_logits, dim=1)
    for i in [1]:
        tp = torch.sum(y_true[:, i, :] * y_pred_logits[:, i, :], dim=1)
        fp = torch.sum((1 - y_true[:, i, :]) * y_pred_logits[:, i, :], dim=1)
        fn = torch.sum(y_true[:, i, :] * (1 - y_pred_logits[:, i, :]), dim=1)
        f1 = (2 * tp + 1e-9) / (2 * tp + fp + fn + 1e-7)
        f1_loss += (1 - f1).mean()
    
    return weight_bce * ce_loss + weight_usage * mse_usage + weight_f1 * f1_loss

class PangolinLitModule(L.LightningModule):
    def __init__(self, model_type='exp', L=L_DIM, W=None, AR=None, switch_epoch=25):
        super().__init__()
        self.save_hyperparameters()

        self.model = PangolinEXP(L, W, AR) if model_type == 'exp' else Pangolin(L, W, AR)
        self.switch_epoch = switch_epoch
        self.loss_fn = masked_focal_mse
        self.use_f1 = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#        loss = masked_focal_mse(y_hat, y)
        loss = hybrid_loss(y_hat, y) if self.use_f1 else masked_focal_mse(y_hat, y)
        #loss = hybrid_loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        ## if <1% 
        if not self.use_f1 and self.current_epoch >= self.switch_epoch:
            self.use_f1 = True
            print(f"🔁 Switching to soft F1 loss at epoch {self.current_epoch}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def predict_step(self, batch, batch_idx):
        return self.model(batch)

class PangolinDataModule(L.LightningDataModule):
    def __init__(self, input_path, batch_size=10, model_type="exp"):
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.model_type = model_type

    def setup(self, stage=None):
        data = torch.load(self.input_path, weights_only=True)
        X = torch.stack(data['X'])
        y = torch.stack(data['y'])
        X = X[:, :4, :] if self.model_type == "seq" else X
        self.dataset = TensorDataset(X, y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

def main():
    parser = argparse.ArgumentParser(description="Train Pangolin model on PSI data")
    parser.add_argument('--input', type=str, required=True, help="Path to training data (.pt)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--model', type=str, choices=["seq", "exp"], default="exp",
                        help="Model type: 'seq' for Pangolin, 'exp' for PangolinEXP")
    parser.add_argument('--output', type=str, default="trained_model.pt", help="Output file name for model")
    args = parser.parse_args()

    W = np.asarray([11]*8 + [21]*4 + [41]*4)
    AR = np.asarray([1]*4 + [4]*4 + [10]*4 + [25]*4)

    dm = PangolinDataModule(input_path=args.input, model_type=args.model)
    model = PangolinLitModule(model_type=args.model, W=W, AR=AR)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(args.output) or ".",
        filename=os.path.basename(args.output).replace(".ckpt", "_epoch{epoch:02d}"),
        save_top_k=-1,
        every_n_epochs=100
    )

    # WandB logger
    wandb_logger = WandbLogger(project="DL_RNA_Splicing", name=os.path.basename(args.output))

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accelerator="auto"
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    main()