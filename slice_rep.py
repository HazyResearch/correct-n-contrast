"""
Alternative ERM model predictions by clustering representations
"""
import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image
from itertools import permutations
from tqdm import tqdm

# Representation-based slicing
import umap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Use a scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Data
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from datasets import train_val_split, get_resampled_indices, get_resampled_set

# Logging and training
from slice import compute_pseudolabels, train_spurious_model, compute_slice_indices
from utils.logging import log_data, initialize_csv_metrics
from train import train_model, test_model, train, evaluate
from utils import print_header, init_experiment

from utils.logging import summarize_acc, log_data
from utils.visualize import plot_confusion, plot_data_batch

# Model
from network import get_net, get_optim, get_criterion, save_checkpoint
from activations import save_activations


def compute_slice_indices_by_rep(model, dataloader,
                                 cluster_umap=True, 
                                 umap_components=2,
                                 cluster_method='kmeans',
                                 args=None,
                                 visualize=False,
                                 cmap='tab10'):
    embeddings, predictions = save_activations(model, 
                                               dataloader, 
                                               args)
    if cluster_umap:
        umap_ = umap.UMAP(random_state=args.seed, 
                      n_components=umap_components)
        X = umap_.fit_transform(embeddings)
    else:
        X = embeddings
    n_clusters = args.num_classes
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters,
                           random_state=args.seed,
                           n_init=10)
        cluster_labels = clusterer.fit_predict(X)
        means = clusterer.cluster_centers_
    elif cluster_method == 'gmm':
        clusterer = GaussianMixture(n_components=n_clusters,
                                    random_state=args.seed,
                                    n_init=10)
        cluster_labels = clusterer.fit_predict(X)
        means = clusterer.means_
    else:
        raise NotImplementedError
    # Match clustering labels to training set    
    cluster_labels, cluster_correct = compute_cluster_assignment(cluster_labels, 
                                                                 dataloader)
    sliced_data_indices = []
    sliced_data_correct = []
    sliced_data_losses = []  # Not actually losses, but distance from point to cluster mean
    for label in np.unique(cluster_labels):
        group = np.where(cluster_labels == label)[0]
        sliced_data_indices.append(group)
        sliced_data_correct.append(cluster_correct[group])
        center = means[label]
        l2_dist = np.linalg.norm(X[group] - center, axis=1)
        sliced_data_losses.append(l2_dist)
    if visualize:
        colors = np.array(cluster_labels).astype(int)
        num_colors = len(np.unique(colors))
        plt.scatter(X[:, 0], X[:, 1], c=colors, s=1.0,
                    cmap=plt.cm.get_cmap(cmap, num_colors))
        plt.colorbar(ticks=np.unique(colors))
        fpath = os.path.join(args.image_path,
                             f'umap-init_slice-cr-{args.experiment_name}.png')
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f'Saved UMAP to {fpath}!')
        
        # Save based on other info too
        targets_all = dataloader.dataset.targets_all
        for target_type in ['target', 'spurious']:
            colors = np.array(targets_all[target_type]).astype(int)
            num_colors = len(np.unique(colors))
            plt.scatter(X[:, 0], X[:, 1], c=colors, s=1.0,
                        cmap=plt.cm.get_cmap(cmap, num_colors))
            plt.colorbar(ticks=np.unique(colors))
            t = f'{target_type[0]}{target_type[-1]}'
            fpath = os.path.join(args.image_path,
                                 f'umap-init_slice-{t}-{args.experiment_name}.png')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
            print(f'Saved UMAP to {fpath}!')
            plt.close()
    return sliced_data_indices, sliced_data_correct, sliced_data_losses


def compute_cluster_assignment(cluster_labels, dataloader):
    all_correct = []
    all_correct_by_datapoint = []
    all_targets = dataloader.dataset.targets_all['target']
    
    # This permutations thing is gross - not actually Hungarian here?
    cluster_label_permute = list(permutations(np.unique(cluster_labels)))
    for cluster_map in cluster_label_permute:
        preds = np.vectorize(cluster_map.__getitem__)(cluster_labels)
        all_targets
        correct = (preds == all_targets)
        all_correct.append(correct.sum())
        all_correct_by_datapoint.append(correct)
    all_correct = np.array(all_correct) / len(all_targets)
    
    # Find best assignment
    best_map = cluster_label_permute[np.argmax(all_correct)]
    cluster_labels = np.vectorize(best_map.__getitem__)(cluster_labels)
    cluster_correct = all_correct_by_datapoint[
        np.argmax(all_correct)].astype(int)
    return cluster_labels, cluster_correct


def combine_data_indices(sliced_data_indices, sliced_data_correct):
    """
    If computing slices from both the ERM model's predictions and 
    representation clustering, use to consolidate into single list of slice indices
    Args:
    - sliced_data_indices (np.array[][]): List of list of sliced indices from ERM and representation clustering, 
                                          e.g. [sliced_indices_pred, sliced_indices_rep],
                                          where sliced_indices_pred = [indices_with_pred_val_1, ... indices_with_pred_val_N]
    - sliced_data_correct (np.array[][]): Same as above, but if the prediction / cluster assignment was correct
    Returns:
    - total_sliced_data_indices (np.array[]): List of combined data indices per slice
    - total_sliced_data_correct (np.array[]): List of combined per-data losses per slice
    """
    sliced_data_indices, sliced_data_indices_ = sliced_data_indices
    sliced_data_correct, sliced_data_correct_ = sliced_data_correct
    total_sliced_data_indices = [[i] for i in sliced_data_indices]
    total_sliced_data_correct = [[c] for c in sliced_data_correct]
    for slice_ix, indices in enumerate(sliced_data_indices_):
        incorrect_ix = np.where(sliced_data_correct_[slice_ix] == 0)[0]
        incorrect_ix_rep = np.where(total_sliced_data_correct[slice_ix][0] == 0)[0]
        incorrect_indices = []
        # This may be slow?
        for i in indices[incorrect_ix]:
            if i not in total_sliced_data_indices[slice_ix][0][incorrect_ix_rep]:
                incorrect_indices.append(i)
        total_sliced_data_indices[slice_ix].append(np.array(incorrect_indices).astype(int))
        total_sliced_data_correct[slice_ix].append(np.zeros(len(incorrect_indices)))
        total_sliced_data_indices[slice_ix] = np.concatenate(total_sliced_data_indices[slice_ix])
        total_sliced_data_correct[slice_ix] = np.concatenate(total_sliced_data_correct[slice_ix])
    return total_sliced_data_indices, total_sliced_data_correct
    