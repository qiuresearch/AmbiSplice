import os
import numpy as np
import torch
import torch.nn.functional as F
# from scipy import integrate
from AmbiSplice import utils
ilogger = utils.get_pylogger(os.path.basename(__name__))

def topks_roc_prc_metrics(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=False):
    """ Compute top-k accuracy, precision, recall, f1, auroc, auprc
        y_pred: 1D or 2D numpy array of predicted scores (probabilities)
        y_true: 1D or 2D numpy array of ground truth binary labels (0 or 1)
        ks: list of k values to compute top-k metrics
        multiples_of_true: if True, k is interpreted as multiples of the number of positive samples in y_true
    """

    if y_true.ndim > 1:
        y_true = y_true.flatten()
    idx_true = torch.where(y_true > 0)[0]
    if idx_true.size == 0:
        ilogger.warning("No positive samples in y_true.")
        return None

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    argsorted_y_pred = torch.argsort(y_pred)  # ascending order
    sorted_y_pred = y_pred[argsorted_y_pred]

    all_metrics = {'k0': ks}

    if multiples_of_true:
        ks = [int(np.ceil(k * len(idx_true))) for k in ks]
    else:
        ks = [int(np.ceil(k)) for k in ks]

    for k in ks:
        metric = {}
        if k < 1 or k > len(y_true):
            continue

        if k * len(idx_true) < len(y_true):
            # tp = torch.intersect1d(argsorted_y_pred[-k:], idx_true)
            tp = torch.tensor(len(idx_true) + k - len(torch.cat([idx_true, argsorted_y_pred[-k:]]).unique()))
        else:
            tp = y_true[argsorted_y_pred[-k:]].sum()

        metric['k'] = k
        metric['topk_threshold'] = sorted_y_pred[-k-1:-k+1].mean() if k > 1 else sorted_y_pred[-1]
        metric['topk_accuracy'] = tp / min(k, len(idx_true))
        metric['topk_precision'] = tp / k
        metric['topk_recall'] = tp / len(idx_true)

        if metric['topk_precision'] + metric['topk_recall'] > 0:
            metric['topk_f1'] = 2 * metric['topk_precision'] * metric['topk_recall'] / (metric['topk_precision'] + metric['topk_recall'])
        else:
            metric['topk_f1'] = 0.0

        for key in metric:
            if key in all_metrics:
                all_metrics[key].append(metric[key])
            else:
                all_metrics[key] = [metric[key]]

    # convert all metrics to numpy arrays (HaHa)
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])

    # auprc
    idx_thresholds = np.linspace(0, len(y_pred) - 1, num=min(len(y_true), 101), dtype=int)
    idx_linspace = np.array([np.searchsorted(sorted_y_pred, t, side='left') for t in np.linspace(0.01, 0.99, num=99)])
    idx_linspace = idx_linspace[(idx_linspace > 0) & (idx_linspace < len(y_pred))]
    
    idx_thresholds = np.sort(np.unique(np.concatenate((idx_thresholds, idx_linspace))))
    thresholds = np.concatenate([sorted_y_pred[idx_thresholds], [1.0]])
    idx_thresholds = np.concatenate([idx_thresholds, [len(y_true)]])

    tprs = np.zeros_like(thresholds)
    fprs = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    recalls = np.zeros_like(thresholds)

    for i, idx in enumerate(idx_thresholds):
        tp = y_true[argsorted_y_pred[idx:]].sum()
        fp = len(y_true) - idx - tp
        fn = len(idx_true) - tp

        tprs[i] = tp / len(idx_true)
        fprs[i] = fp / (len(y_true) - len(idx_true))

        if tp + fp > 0:
            precisions[i] = tp / (tp + fp)
        else:
            precisions[i] = 1.0
        if tp + fn > 0:
            recalls[i] = tp / (tp + fn)
        else:
            recalls[i] = 0.0

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    sorted_indices = np.arange(len(recalls)) # [np.argsort(recalls)]
    # compute AUC using the trapezoidal rule with numpy
    roc_auc = -np.trapz(tprs[sorted_indices], fprs[sorted_indices])
    prc_auc = -np.trapz(precisions[sorted_indices], recalls[sorted_indices])
    # alternatively, we can also use scipy.integrate to compute AUC
    # roc_auc = integrate.trapz(tprs[sorted_indices], fprs[sorted_indices])
    # prc_auc = integrate.trapz(precisions[sorted_indices], recalls[sorted_indices])

    all_metrics.update({
        'idx_threshold': idx_thresholds,
        'threshold': thresholds[sorted_indices],
        'tpr': tprs[sorted_indices],
        'fpr': fprs[sorted_indices],
        'recall': recalls[sorted_indices],
        'precision': precisions[sorted_indices],
        'f1': f1s[sorted_indices],
        'auprc': prc_auc,
        'auroc': roc_auc,
    })

    return all_metrics

def calc_loss(preds, labels):
    """ 
        labels are the ground truth in the batch_feats
    """
    # TODO::
    # 1) Many sequences are shorter than crop_size, so we need to mask out the padded regions
    # cls has shape (B, 4, crop_size)
    # psi has shape (B, 1, crop_size)
    # 2) cls_odds and psi_std are not used currently

    loss_items = {}
    if 'cls' not in labels or 'psi' not in labels:
        return -1, loss_items
    
    # four classes: non, acceptor, donor, hybrid
    if 'cls_logits' in preds:
        loss_items['cls_loss'] = F.cross_entropy(
            preds['cls_logits'],
            labels['cls'],
            ignore_index=-100)
    elif 'cls' in preds:
        loss_items['cls_loss'] = F.cross_entropy(
            torch.log(preds['cls'] + 1e-8),  # add a small value to avoid log(0)
            labels['cls'],
            ignore_index=-100)
    else:
        raise ValueError("No cls or cls_logits in preds.")

    # psi is a probability between 0 and 1
    if 'psi_logits' in preds:
        loss_items['psi_loss'] = F.binary_cross_entropy_with_logits(
            preds['psi_logits'],
            labels['psi'],
            weight=labels['psi'] >= 0,  # mask out negative values
            reduction='mean')
    elif 'psi' in preds:
        psi_label = labels['psi'].float()
        weight = (psi_label >= 0).float()  # mask out negative values
        psi_label[psi_label < 0] = 0  # set negative values to 0 for BCE loss (an error will be raised otherwise)

        loss_items['psi_loss'] = F.binary_cross_entropy(
            preds['psi'].float(),
            psi_label,
            weight=weight,
            reduction='mean')
    else:
        raise ValueError("No psi or psi_logits in preds.")

    loss = loss_items['cls_loss'] + loss_items['psi_loss']
    loss_items['loss'] = loss

    for k in loss_items: # do not move to cpu here, let the caller do it
        loss_items[k] = loss_items[k].detach() 

    return loss, loss_items


def calc_metric(preds, labels, keep_batchdim=False, to_numpy=True, eps=1e-8):
    """ preds are the output of forward() without any activation
        labels are the ground truth in the batch_feats
    """
    metric_items = {}
    if 'cls' not in labels or 'psi' not in labels:
        return metric_items

    # compute precision, recall, f1 for cls
    # cls_logits in (B, num_classes, crop_size)
    if 'cls' in preds:
        cls_pred = torch.movedim(preds['cls'], 1, -1)
    elif 'cls_logits' in preds:
        cls_pred = F.softmax(torch.movedim(preds['cls_logits'], 1, -1), dim=-1)  # (B, crop_size, num_classes)
    else:
        raise ValueError("No cls or cls_logits in preds.")
        
    cls_label = F.one_hot(labels['cls'], num_classes=cls_pred.shape[-1]).to(cls_pred.dtype)  # (B, crop_size, num_classes)

    if keep_batchdim:
        dims_to_sum = tuple(range(1, cls_pred.ndim - 1))  # sum over all but batch and last dimension
    else:
        dims_to_sum = tuple(range(cls_pred.ndim - 1))  # sum over all but last dimension

    # print(cls_pred.shape, cls_label.shape, dims_to_sum)

    cls_tp = (cls_pred * cls_label).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)
    cls_fp = (cls_pred * (1 - cls_label)).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)
    cls_fn = ((1 - cls_pred) * cls_label).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)

    cls_precision = cls_tp / (cls_tp + cls_fp + eps)
    cls_recall = cls_tp / (cls_tp + cls_fn + eps)
    cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + eps)

    if keep_batchdim:
        dims_to_mean = tuple(range(1, cls_f1.ndim))  # average over all but batch dimension
    else:
        dims_to_mean = tuple(range(cls_f1.ndim))  # average over all dimensions

    metric_items['cls_precision'] = cls_precision.mean(dim=dims_to_mean)
    metric_items['cls_recall'] = cls_recall.mean(dim=dims_to_mean)
    metric_items['cls_f1'] = cls_f1.mean(dim=dims_to_mean)

    # compute mse for psi
    if 'psi' in preds:
        psi_pred = preds['psi']
    elif 'psi_logits' in preds:
        psi_pred = torch.sigmoid(preds['psi_logits'])  # (B, crop_size)
    else:
        raise ValueError("No psi or psi_logits in preds.")

    metric_items['psi_mse'] = F.mse_loss(psi_pred, labels['psi'], reduction='none' if keep_batchdim else 'mean')
    if keep_batchdim:
        metric_items['psi_mse'] = metric_items['psi_mse'].mean(dim=tuple(range(1, metric_items['psi_mse'].ndim)))

    if to_numpy:
        for key in metric_items:
            metric_items[key] = metric_items[key].cpu().numpy()

    return metric_items


def calc_benchmark(preds, labels, keep_batchdim=True, eps=1e-8):
    """ preds are the output of forward() without any activation
        labels are the ground truth in the batch_feats
        
        Caution: the channel dimension of preds['cls'] should be the second dimension
    """
    ilogger.info("Calculating loss metrics...")
    loss, benchmark_metrics = calc_loss(preds, labels)
    ilogger.info("Calculating performance metrics...")
    benchmark_metrics.update(calc_metric(preds, labels, keep_batchdim=keep_batchdim, to_numpy=True, eps=eps))

    ilogger.info("Calculating ROC and PRC metrics...")
    if keep_batchdim:
        batch_size = labels['cls'].shape[0]
        for i in range(batch_size):
            y_pred = (preds['cls'][i].argmax(dim=1).flatten() > 0.5).to(torch.float32)
            y_true = (labels['cls'][i].flatten() > 0.5).to(torch.float32)
            metric = topks_roc_prc_metrics(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=True)
            if metric is None:
                continue
            # only collect scalar metrics
            for key in metric:
                if hasattr(metric[key], '__len__') and len(metric[key]) > 1:
                    continue
                if key in benchmark_metrics:
                    benchmark_metrics[key].append(metric[key])
                else:
                    benchmark_metrics[key] = [metric[key]]
        # convert lists to numpy arrays
        for key in benchmark_metrics:
            if isinstance(benchmark_metrics[key], list):
                benchmark_metrics[key] = np.array(benchmark_metrics[key])
    else:
        y_pred = (preds['cls'].argmax(dim=1).flatten() > 0.5).to(torch.float32)
        y_true = (labels['cls'].flatten() > 0.5).to(torch.float32)
        benchmark_metrics.update(topks_roc_prc_metrics(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=True))

    return benchmark_metrics


def save_summary_metrics(metrics, save_path):
    """ Save summary metrics to a yaml file.
        metrics: dict containing summary metrics
        save_path: path to save the yaml file
    """
    pass


def save_individual_metrics(metrics, save_path):
    """ Save individual metrics to a CSV file without Pandas.
        metrics: list of dicts, each dict contains metrics for a batch
        save_path: path to save the CSV file
    """
    keys = list(metrics.keys())
    csv_lines = [','.join(keys)]
    for i in range(len(metrics[keys[0]])):
        csv_lines.append(','.join([str(metrics[key][i]) for key in keys]))

    with open(save_path, 'w') as f:
        f.writelines('\n'.join(csv_lines))    
    return
