"""
CivilComments Dataset
- Code mostly from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/civilcomments_dataset.py
- See WILDS, https://wilds.stanford.edu for more
"""
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

from datasets.grouper import CombinatorialGrouper


class CivilComments(Dataset):
    """
    CivilComments dataset
    """
    def __init__(self, root_dir, target_name='toxic', 
                 confounder_names=['identities'],
                 split='train', transform=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.transform = transform
        
        # Labels
        self.class_names = ['non_toxic', 'toxic']
        
        # Set up data directories
        self.data_dir = os.path.join(self.root_dir)
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')
        
        # Read in metadata
        type_of_split = self.target_name.split('_')[-1]
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'all_data_with_identities.csv'),
            index_col=0)
        
        # Get split
        self.split_array = self.metadata_df['split'].values
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == split]
        
        # Get the y values
        self.y_array = torch.LongTensor(
            self.metadata_df['toxicity'].values >= 0.5)
        self.y_size = 1
        self.n_classes = 2
        
        # Get text
        self.x_array = np.array(self.metadata_df['comment_text'])
        
        # Get confounders and map to groups
        self._identity_vars = ['male',
                               'female',
                               'LGBTQ',
                               'christian',
                               'muslim',
                               'other_religions',
                               'black',
                               'white']
        self._auxiliary_vars = ['identity_any',
                                'severe_toxicity',
                                'obscene',
                                'threat',
                                'insult',
                                'identity_attack',
                                'sexual_explicit']
        self.metadata_array = torch.cat(
            (torch.LongTensor((self.metadata_df.loc[:, self._identity_vars] >= 0.5).values),
             torch.LongTensor((self.metadata_df.loc[:, self._auxiliary_vars] >= 0.5).values),
             self.y_array.reshape((-1, 1))), dim=1)
        
        self.metadata_fields = self._identity_vars + self._auxiliary_vars + ['y']
        self.confounder_array = self.metadata_array[:, np.arange(len(self._identity_vars))]
        self.metadata_map = None
        
        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars]
        
        # Below is nonstandard
        
        # Get sub_targets / group_idx
        groupby_fields = self._identity_vars + ['y']
        self.eval_grouper = CombinatorialGrouper(self, groupby_fields)
        self.group_array = self.eval_grouper.metadata_to_group(self.metadata_array,
                                                               return_counts=False)
        self.n_groups = len(np.unique(self.group_array))  # self.eval_grouper._n_groups
        
        # Get spurious labels
        self.spurious_grouper = CombinatorialGrouper(self, 
                                                     self._identity_vars)
        self.spurious_array = self.spurious_grouper.metadata_to_group(
            self.metadata_array, return_counts=False).numpy()
        
        # Get consistent label attributes
        self.targets = self.y_array
        
        unique_group_ix = np.unique(self.spurious_array)
        group_ix_to_label = {}
        for i, gix in enumerate(unique_group_ix):
            group_ix_to_label[gix] = i
        spurious_labels = [group_ix_to_label[int(s)] 
                           for s in self.spurious_array]
#         spurious_labels = ['_'.join([str(x) for x in
#                                      self.confounder_array[ix]]) 
#                            for ix in range(self.confounder_array.shape[0])]
        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(spurious_labels),
                            'sub_target': np.array(self.metadata_array[:, self.eval_grouper.groupby_field_indices]),
                            'metadata': np.array(self.metadata_array)}
        self.group_labels = [self.group_str(i) for i in range(self.n_groups)]
        
    def __len__(self):
        return len(self.y_array)
    
    def __getitem__(self, idx):
        x = self.x_array[idx]
        y = self.y_array[idx]
        if self.transform is not None:
            x = self.transform(x)
        return (x, y, idx) # g
    
    def group_str(self, group_idx):
        return self.eval_grouper.group_str(group_idx)
    
    def get_text(self, idx):
        return self.x_array[idx]
    

def init_bert_transform(tokenizer, model_name, args):
    """
    Inspired from the WILDS dataset: 
    - https://github.com/p-lambda/wilds/blob/main/examples/transforms.py
    """
    def transform(text):
        tokens = tokenizer(text, padding='max_length', 
                           truncation=True,
                           max_length=args.max_token_length,  # 300
                           return_tensors='pt')
        if model_name == 'bert-base-uncased':
            x = torch.stack((tokens['input_ids'],
                             tokens['attention_mask'],
                             tokens['token_type_ids']), dim=2)
        # Not supported for now
        elif model_name == 'distilbert-base-uncased':
            x = torch.stack((tokens['input_ids'], 
                             tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x
    return transform
    

def load_civilcomments(args, train_shuffle=True):
    """
    Actually load CivilComments
    """
    pretrained_name = args.arch if args.arch[-3:] != '_pt' else args.arch[:-3]
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_name)  # 'bert-base-uncased'
    transform = init_bert_transform(tokenizer, pretrained_name, args)
    
    train_set = CivilComments(args.root_dir, target_name='toxic',
                              confounder_names=['identities'],
                              split='train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=train_shuffle,
                              num_workers=args.num_workers)
    
    val_set = CivilComments(args.root_dir, target_name='toxic',
                            confounder_names=['identities'],
                            split='val', transform=transform)
    val_loader = DataLoader(train_set, batch_size=args.bs_val,
                            shuffle=False, num_workers=args.num_workers)
    
    test_set = CivilComments(args.root_dir, target_name='toxic',
                             confounder_names=['identities'],
                             split='test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.bs_val,
                             shuffle=False, num_workers=args.num_workers)
    
    args.num_classes = 2
    return (train_loader, val_loader, test_loader)


# Refactor for modularity
def load_dataloaders(args, train_shuffle=True, transform=None):
    return load_civilcomments(args, train_shuffle)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    """
    Does not currently apply to NLP datasets
    """
    return 