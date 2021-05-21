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


# python train_supervised_contrast_2.py --arch resnet50_pt --dataset isic --slice_with rep --rep_cluster_method gmm --pretrained_spurious_path "./model/isic/config/cp-a=resnet50_pt-d=isic-tm=2s2s_spur-sc-me=100-bst=32-lr=0.0001-mo=0.9-wd=1.0--spur-bs_trn=32-lr=0.0001-mo=0.9-wd=1.0-rc=subsample-s=0-cpe=9-cpb=0.pth.tar" --num_positive 64 --num_negative 64 --num_anchor 64 --batch_factor 32 --train_encoder --target_sample_ratio 1.0 --temperature 0.1 --lr 1e-4 --momentum 0.9 --weight_decay 1e-3 --stopping_window 32 --log_loss_interval 10 --checkpoint_interval 100000 --log_visual_interval 400000 --verbose --no_projection_head --contrastive_weight 0.75 -cs apn --seed 0 --replicate 0 --num_negative_easy 0 --max_epoch 100 --resample_class subsample --replicate 12 --num_negative_easy 64