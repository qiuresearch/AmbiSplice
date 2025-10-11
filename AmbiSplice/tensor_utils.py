import os
import yaml
import numpy as np
import torch

def to_yaml(in_data, yaml_path=None):

    def _to_yaml(in_data):
        if isinstance(in_data, (list, tuple)):
            return in_data
        elif isinstance(in_data, np.ndarray):
            return in_data.tolist()
        elif isinstance(in_data, np.generic):
            return in_data.item()
        elif isinstance(in_data, torch.Tensor):
            return in_data.cpu().numpy().tolist()
        elif isinstance(in_data, dict):
            return {k: _to_yaml(v) for k, v in in_data.items()}
        else:
            return in_data

    yaml_data = _to_yaml(in_data)
    if yaml_path is not None:
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)
    return yaml_data


def concat_dicts_outputs(epoch_outputs):
    """ Concatenate list of batch outputs to single batched outputs."""
    # initialize a new dictionary to hold all outputs
    dataset_feats, dataset_preds = {}, {}
    for batch_feats, batch_preds in epoch_outputs:
        for key, value in batch_feats.items():
            if key not in dataset_feats:
                dataset_feats[key] = []
            dataset_feats[key].append(value)
        for key, value in batch_preds.items():
            if key not in dataset_preds:
                dataset_preds[key] = []
            dataset_preds[key].append(value)

    # concatenate lists into single tensors if they are tensors or numpy arrays
    def _concat_lists_of_dict(dict_of_lists):
        for key in dict_of_lists:
            if isinstance(dict_of_lists[key][0], torch.Tensor):
                dict_of_lists[key] = torch.cat(dict_of_lists[key], dim=0)
            elif isinstance(dict_of_lists[key][0], np.ndarray):
                dict_of_lists[key] = np.concatenate(dict_of_lists[key], axis=0)
            elif isinstance(dict_of_lists[key][0], (list, tuple)):
                dict_of_lists[key] = sum(dict_of_lists[key], start=[])
            else:
                continue
        return dict_of_lists

    dataset_feats = _concat_lists_of_dict(dataset_feats)
    dataset_preds = _concat_lists_of_dict(dataset_preds)

    return dataset_feats, dataset_preds
