"""
Datasets
"""
import copy
import numpy as np
import importlib


def initialize_data(args):
    """
    Set dataset-specific arguments
    By default, the args.root_dir below should work ifinstalling datasets as
    specified in the README to the specified locations
    - Otherwise, change `args.root_dir` to the path where the data is stored.
    """
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')
    
    if 'waterbirds' in args.dataset:
        args.root_dir = './datasets/data/Waterbirds/'
        args.root_dir = '/dfs/scratch1/mzhang/projects/slice-and-dice-smol/datasets/data/Waterbirds'
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ['landbirds', 'waterbirds']
        if args.dataset == 'waterbirds_r':
            args.train_classes = ['land', 'water']
            
    elif 'colored_mnist' in args.dataset:
        args.root_dir = './datasets/data/'
        args.data_path = './datasets/data/'
        args.target_name = 'digit'
        args.confounder_names = ['color']
        args.image_mean = 0.5
        args.image_std = 0.5
        args.augment_data = False
        # args.train_classes = args.train_classes
    
    elif 'celebA' in args.dataset:
        args.root_dir = '/dfs/scratch0/nims/CelebA/celeba/' 
        # args.root_dir = './datasets/data/CelebA/'
        # IMPORTANT - dataloader assumes that we have directory structure
        # in ./datasets/data/CelebA/ :
        # |-- list_attr_celeba.csv
        # |-- list_eval_partition.csv
        # |-- img_align_celeba/
        #     |-- image1.png
        #     |-- ...
        #     |-- imageN.png
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['blond', 'nonblond']
        args.val_split = 0.2
        
    elif 'civilcomments' in args.dataset:
        args.root_dir = './datasets/data/CivilComments/'
        args.target_name = 'toxic'
        args.confounder_names = ['identities']
        args.image_mean = 0
        args.image_std = 0
        args.augment_data = False
        args.image_path = './images/civilcomments/'
        args.train_classes = ['non_toxic', 'toxic']
        args.max_token_length = 300
    
    args.task = args.dataset  # e.g. 'civilcomments', for BERT
    args.num_classes = len(args.train_classes)
    return load_dataloaders, visualize_dataset


def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def get_resampled_indices(dataloader, args, sampling='subsample', seed=None):
    """
    Args:
    - dataloader (torch.utils.data.DataLoader): 
    - sampling (str): 'subsample' or 'upsample'
    """
    try:
        indices = dataloader.sampler.indices
    except:
        indices = np.arange(len(dataloader.dataset))
    indices = np.arange(len(dataloader.dataset))
    target_vals, target_val_counts = np.unique(
        dataloader.dataset.targets_all['target'][indices], 
        return_counts=True)
    sampled_indices = []
    if sampling == 'subsample':
        sample_size = np.min(target_val_counts)
    elif sampling == 'upsample':
        sample_size = np.max(target_val_counts)
    else:
        return indices
        
    if seed is None:
        seed = args.seed
    np.random.seed(seed)
    for v in target_vals:
        group_indices = np.where(
            dataloader.dataset.targets_all['target'][indices] == v)[0]
        if sampling == 'subsample':
            sampling_size = np.min([len(group_indices), sample_size])
            replace = False
        elif sampling == 'upsample':
            sampling_size = np.max([0, sample_size - len(group_indices)])
            sampled_indices.append(group_indices)
            replace = True
        sampled_indices.append(np.random.choice(
            group_indices, size=sampling_size, replace=replace))
    sampled_indices = np.concatenate(sampled_indices)
    np.random.seed(seed)
    np.random.shuffle(sampled_indices)
    return indices[sampled_indices]


def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain spurious dataset resampled_set
    Args:
    - dataset (torch.utils.data.Dataset): Spurious correlations dataset
    - resampled_set_indices (int[]): List-like of indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:  # Some dataset classes may not have these attributes
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.y_array
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except:
            resampled_set_indices = np.concatenate(resampled_set_indices)
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        try:
            resampled_set.df = resampled_set.df.iloc[resampled_set_indices]
        except AttributeError:
            pass
            
        try:
            resampled_set.data = resampled_set.data[resampled_set_indices]
        except AttributeError:
            pass
    
    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]
    return resampled_set
