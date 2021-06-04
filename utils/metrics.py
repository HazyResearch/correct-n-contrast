"""
Functions for computing useful metrics, e.g. entropy, conditional entropy
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def compute_entropy(targets):
    vals, counts = np.unique(targets, return_counts=True)
    probs = counts / len(targets)
    return -1 * np.sum([p * np.log(p) for p in probs])


def log_label_mutual_info(sliced_data_indices, ):
    print(f'len(sliced_data_indices): {len(sliced_data_indices)}')
    # Report empirical MI(Y | Z_s) = \sum_{z_s} (H(Y) - H(Y | Z_s = z_s))
    print_header('Resampled MI', style='top')
    mi_by_slice = compute_mutual_info_by_slice(train_loader,
                                               sliced_data_indices)
    for ix, mi in enumerate(mi_by_slice):
        print(f'H(Y) - H(Y | Z = z_{ix}) = {mi:<.3f} (by slice)')
    mi_resampled = compute_resampled_mutual_info(train_loader,
                                                 sliced_data_indices)
    print_header(f'H(Y) - H(Y | Z) = {mi_resampled:<.3f}')
    args.mi_resampled = mi_resampled


def compute_mutual_info_by_slice(dataloader, sliced_data_indices):
    mutual_info_by_slice = []
    for indices in sliced_data_indices:
        slice_targets = dataloader.dataset.targets_all['target'][indices]
        slice_spurious = dataloader.dataset.targets_all['spurious'][indices]
        entropy_y = compute_entropy(slice_targets)
        conditional_entropies = []
        slice_vals, slice_counts = np.unique(slice_spurious, return_counts=True)
        # For now, report only for max one
        max_val_ix = np.argmax(slice_counts)
        for ix, slice_val in enumerate([slice_vals[max_val_ix]]):
            conditional_indices = np.where(slice_spurious == slice_val)[0]
            cond_entropy = compute_entropy(slice_targets[conditional_indices])
            conditional_entropies.append(cond_entropy)
        mutual_info_by_slice.append(np.mean([entropy_y - c_e for c_e in conditional_entropies]))
    return mutual_info_by_slice


def compute_resampled_mutual_info(dataloader, sliced_data_indices):
    indices = np.hstack(sliced_data_indices)
    slice_targets = dataloader.dataset.targets_all['target'][indices]
    slice_spurious = dataloader.dataset.targets_all['spurious'][indices]
    entropy_y = compute_entropy(slice_targets)
    conditional_entropies = []
    slice_vals, slice_counts = np.unique(slice_spurious, return_counts=True)
    for ix, slice_val in enumerate(slice_vals):
        conditional_indices = np.where(slice_spurious == slice_val)[0]
        cond_entropy = compute_entropy(slice_targets[conditional_indices])
        conditional_entropies.append(cond_entropy)
    return np.mean([entropy_y - c_e for c_e in conditional_entropies])


def compute_roc_auc(targets, probs):
    """'Safe' AUROC computation"""
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.numpy()
    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = -1
    return auroc
