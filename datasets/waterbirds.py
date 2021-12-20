"""
Waterbirds Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more details
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.models import model_attributes
from utils.visualize import plot_data_batch
from copy import deepcopy


class Waterbirds(Dataset):
    """
    Waterbirds dataset from waterbird_complete95_forest2water2 in GroupDRO paper
    """

    def __init__(self, root_dir, target_name, confounder_names,
                 split, augment_data=False, model_type=None, args=None,
                 train_transform=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        if '_pt' in model_type:
            self.model_type = model_type[:-3]
        self.augment_data = augment_data
        self.split = split

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.data_dir = os.path.join(
            self.root_dir,
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        # Filter for data split ('train', 'val', 'test')
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1

        # Reverse
        if args.dataset == 'waterbirds_r':
            self.y_array = self.metadata_df['place'].values
            self.confounder_array = self.metadata_df['y'].values

        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values

        # Play nice with my earlier code
        self.targets = torch.tensor(self.y_array)
        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(self.confounder_array),
                            'sub_target': np.array(list(zip(self.y_array, self.confounder_array)))}
        self.group_labels = ['LANDBIRD on land', 'LANDBIRD on water',
                             'WATERBIRD on land', 'WATERBIRD on water']

        if args.dataset == 'waterbirds_r':
            self.group_labels = ['LAND with landbird', 'LAND with waterbird',
                                 'WATER with landbird', 'WATER with waterbird']

        # Set transform
        if model_attributes[self.model_type]['feature_type'] == 'precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
            # Added for 
            self.data = self.features_mat
        else:
            self.features_mat = None
            if train_transform is None:
                self.train_transform = get_transform_cub(
                    self.model_type,
                    train=True,
                    augment_data=augment_data)
            else:
                self.train_transform = train_transform
            self.eval_transform = get_transform_cub(
                self.model_type,
                train=False,
                augment_data=augment_data)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.targets[idx]  # changed to fit with earlier code
        # g = self.group_array[idx]
        if model_attributes[self.model_type]['feature_type'] == 'precomputed':
            x = self.features_mat[idx, :]
            print('loading from features_mat')
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
                  self.eval_transform):
                img = self.eval_transform(img)
            # Flatten if needed
            if model_attributes[self.model_type]['flatten']:
                assert img.dim() == 3
                img = img.view(-1)
            x = img

        return (x, y, idx)

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0 / 224.0
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def load_waterbirds(args, train_shuffle=True, transform=None):
    """
    Default dataloader setup for Waterbirds

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = Waterbirds(args.root_dir,
                           target_name=args.target_name,
                           confounder_names=args.confounder_names,
                           split='train', model_type=args.arch,
                           args=args, train_transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=train_shuffle,
                              num_workers=args.num_workers)

    val_set = Waterbirds(args.root_dir,
                         target_name=args.target_name,
                         confounder_names=args.confounder_names,
                         split='val', model_type=args.arch, args=args)
    val_loader = DataLoader(val_set, batch_size=args.bs_val,
                            shuffle=False, num_workers=args.num_workers)

    test_set = Waterbirds(args.root_dir,
                          target_name=args.target_name,
                          confounder_names=args.confounder_names,
                          split='test', model_type=args.arch, args=args)
    test_loader = DataLoader(test_set, batch_size=args.bs_val,
                             shuffle=False, num_workers=args.num_workers)
    args.num_classes = 2
    return (train_loader, val_loader, test_loader)


def visualize_waterbirds(dataloader, num_datapoints, title, args, save,
                         save_id, ftype='png', target_type='group_idx'):
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        if subset:
            class_indices = subset_indices[class_indices]
        all_data_indices.extend(class_indices[:num_datapoints])
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in all_data_indices],
                    mean=np.mean([0.485, 0.456, 0.406]),  
                    std=np.mean([0.229, 0.224, 0.225]), nrow=8, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)
    
    
def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain spurious dataset resampled_set
    Args:
    - dataset (torch.utils.data.Dataset): Spurious correlations dataset
    - resampled_set_indices (int[]): List-like of indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
    resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
    resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
    resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
    
    resampled_set.targets = resampled_set.y_array
    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]
    return resampled_set


# Refactor for modularity
def load_dataloaders(args, train_shuffle=True, transform=None):
    return load_waterbirds(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    return visualize_waterbirds(dataloader, num_datapoints, title, 
                                args, save, save_id, ftype, target_type)