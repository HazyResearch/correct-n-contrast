"""
CXR8 Dataset
- Modified from https://github.com/jrzech/reproduce-chexnet
- Modified from https://github.com/nimz/stratification/blob/master/datasets/cxr.py

Example command:
python train_supervised_contrast.py --dataset cxr --arch resnet50_pt --train_encoder --pretrained_spurious_path "" --optim sgd --lr_s 1e-4 --momentum_s 0 --weight_decay_s 1e-4 --bs_trn_s 32 --max_epoch_s 50 --num_anchor 64 --num_positive 64 --num_negative 64 --num_negative_easy 64 --batch_factor 32 --lr 1e-4 --momentum 0.9 --weight_decay 1e-4 --weight_decay_c 1e-4 --target_sample_ratio 1 --temperature 0.05 --max_epoch 15 --no_projection_head --contrastive_weight 0.75 --log_visual_interval 10000 --checkpoint_interval 10000 --verbose --log_loss_interval 10 --replicate 42 --seed 42 --resample_class subsample
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import pydicom  # Loading CXR files
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.visualize import plot_data_batch


class CXR(Dataset):
    """
    CXR8 Dataset
    - Originally from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
    """
    # Image details
    img_channels = 3
    img_resolution = 224
    img_norm = {'mean': (0.48865, 0.48865, 0.48865),
                'std': (0.24621, 0.24621, 0.24621)}
    def __init__(self, root_dir, target_name='pmx', 
                 confounder_names=['chest_tube'],
                 split='train', augment_data=False, 
                 train_transform=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names 
        # Only support 1 confounder for now
        confounder_names = self.confounder_names[0]  
        self.split = split
        
        # Only for the metadata file
        self.data_dir = os.path.join('./datasets/data/CXR',
                                     'cxr_examples-train-rle.csv')
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')
            
        self.metadata_df = pd.read_csv(self.data_dir)
        
        # Gross - get the right split
        if self.split == 'train':
            self.metadata_df =  self.metadata_df[(self.metadata_df['split'] == 'train') & 
                                                 (self.metadata_df['chest_tube'].isnull())]
        elif split == 'val':
            self.metadata_df = self.metadata_df[(self.metadata_df['split'] == 'train') & 
                             (self.metadata_df['chest_tube'] >= 0)]
        elif split == 'test':
            self.metadata_df = self.metadata_df[(self.metadata_df['split'] == 'test')]
            
        # Groundtruths
        self.y_array = self.metadata_df[self.target_name].values

        # Spurious attributes
        self.confounder_array = self.metadata_df[confounder_names].values
        
        ## Training data has no spurious attribute labels, assume no chest tube
        self.confounder_array[np.isnan(self.confounder_array)] = 0  # Assume no chest tubes apriori
        self.n_classes = len(np.unique(self.y_array))
        self.n_confounders = 2  # len(self.confounder_names)
        
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype('int')
        
        self.filename_array = self.metadata_df['filepath'].values
        
        self.targets = torch.tensor(self.y_array)
        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(self.confounder_array),
                            'sub_target': np.array(list(zip(self.y_array, self.confounder_array)))}
        
        self.group_labels = ['NO PMX, no chest tube', 'NO PMX, chest tube',
                             'PMX, no chest tube', 'PMX, chest tube']
        
        if train_transform is None:
            self.train_transform = get_transform_cxr(train=True,
                                                     augment_data=augment_data)
        else:
            self.train_transform = train_transform
        self.eval_transform = get_transform_cxr(train=False,
                                                augment_data=augment_data)
        
    def __len__(self):
        return len(self.y_array)
    
    def __getitem__(self, idx):
        y = self.targets[idx]  # changed to fit with earlier code
        img_filepath = self.filename_array[idx]
        
        ds = pydicom.dcmread(img_filepath)
        img = ds.pixel_array
        img = Image.fromarray(img)
        
        if self.split == 'train':
            img = self.train_transform(img)
        else:
            img = self.eval_transform(img)
        img = img.repeat([3,1,1])
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
    

def get_transform_cxr(train, augment_data):
    """
    Currently both :train: and :augment_data: are dummies
    """
    CXR_MEAN = 0.48865
    CXR_STD = 0.24621
    CXR_SIZE = 224
    transform = transforms.Compose([
        transforms.Resize([CXR_SIZE, CXR_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(CXR_MEAN, CXR_STD),
    ])
    return transform
        
        
def load_cxr(args, train_shuffle=True, transform=None):
    """
    Default dataloader setup for CXR

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = CXR(args.root_dir,
                    target_name=args.target_name,
                    confounder_names=args.confounder_names,
                    split='train', train_transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=train_shuffle, 
                              num_workers=args.num_workers)
    
    val_set = CXR(args.root_dir,
                  target_name=args.target_name,
                  confounder_names=args.confounder_names,
                  split='val', train_transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.bs_trn,
                            shuffle=train_shuffle, 
                            num_workers=args.num_workers)
    
    test_set = CXR(args.root_dir,
                   target_name=args.target_name,
                   confounder_names=args.confounder_names,
                   split='test', train_transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.bs_trn,
                             shuffle=train_shuffle, 
                             num_workers=args.num_workers)

    args.num_classes = 2
    return (train_loader, val_loader, test_loader)


def visualize_cxr(dataloader, num_datapoints, title, args, save,
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

    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in
                     all_data_indices],
                    mean=0.48865, std=0.24621, nrow=8, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)
   

# Refactor for modularity
def load_dataloaders(args, train_shuffle=True, transform=None):
    return load_cxr(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    return visualize_cxr(dataloader, num_datapoints, title, 
                         args, save, save_id, ftype, target_type)
