"""
Functions for slicing data

NOTE: Going to refactor this with slice_train.py and spurious_train.py
      - Currently methods support different demos / explorations
"""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from datasets import train_val_split, get_resampled_set, get_resampled_indices
from train import train_model, test_model
from network import get_criterion, get_optim, get_net, get_output


def compute_slice_indices(net, dataloader, criterion, 
                          batch_size, args, resample_by='class',
                          loss_factor=1., use_dataloader=False):
    """
    Use trained model to predict "slices" of data belonging to different subgroups

    Args:
    - net (torch.nn.Module): Pytorch neural network model
    - dataloader (torch.nn.utils.DataLoader): Pytorch data loader
    - criterion (torch.nn.Loss): Pytorch cross-entropy loss (with reduction='none')
    - batch_size (int): Batch size to compute slices over
    - args (argparse): Experiment arguments
    - resamble_by (str): How to resample, ['class', 'correct']
    Returns:
    - sliced_data_indices (int(np.array)[]): List of numpy arrays denoting indices of the dataloader.dataset
                                             corresponding to different slices
    """
    # First compute pseudolabels
    dataloader_ = dataloader if use_dataloader else None
    dataset = dataloader.dataset
    slice_outputs = compute_pseudolabels(net, dataset, 
                                         batch_size, args,  # Added this dataloader
                                         criterion, dataloader=dataloader_)
    pseudo_labels, outputs, correct, correct_spurious, losses = slice_outputs
    
    output_probabilities = torch.exp(outputs) / torch.exp(outputs).sum(dim=1).unsqueeze(dim=1)

    sliced_data_indices = []
    all_losses = []
    all_correct = []
    correct = correct.detach().cpu().numpy()
    all_probs = []
    for label in np.unique(pseudo_labels):
        group = np.where(pseudo_labels == label)[0]
        if args.weigh_slice_samples_by_loss:
            losses_per_group = losses[group]
        correct_by_group = correct[group]
        probs_by_group = output_probabilities[group]
        if args.subsample_labels is True or args.supersample_labels is True:
            group_vals = np.unique(dataloader.dataset.targets[group],
                                   return_counts=True)[1]
            sample_size = (np.min(group_vals) if args.subsample_labels is True
                           else np.max(group_vals))
            sampled_indices = []
            # These end up being the same
            if resample_by == 'class':
                target_values = dataloader.dataset.targets[group]
            elif resample_by == 'correct':
                target_values = correct_by_group
            # assert correct_by_group == dataloader.dataset.targets[group]
            print(f'> Resampling by {resample_by}...')
            for v in np.unique(target_values):
                group_indices = np.where(target_values == v)[0]
                if args.subsample_labels is True:
                    sampling_size = np.min([len(group_indices), sample_size])
                    replace = False
                    p = None
                elif args.supersample_labels is True:
                    sampling_size = np.max(
                        [0, sample_size - len(group_indices)])
                    sampled_indices.append(group_indices)
                    replace = True
                    if args.weigh_slice_samples_by_loss:
                        p = losses_per_group[group_indices] * loss_factor
                        p = (torch.exp(p) / torch.exp(p).sum()).numpy()
                    else:
                        p = None
                sampled_indices.append(np.random.choice(
                    group_indices, size=sampling_size, replace=replace, p=p)) 
            sampled_indices = np.concatenate(sampled_indices)
            sorted_indices = np.arange(len(sampled_indices))
            if args.weigh_slice_samples_by_loss:
                all_losses.append(losses_per_group[sampled_indices][sorted_indices])
            sorted_indices = np.arange(len(sampled_indices))
            sliced_data_indices.append(group[sampled_indices][sorted_indices])
            all_correct.append(correct_by_group[sampled_indices][sorted_indices])
            all_probs.append(probs_by_group[sampled_indices][sorted_indices])
        else:
            if args.weigh_slice_samples_by_loss:
                sorted_indices = torch.argsort(losses_per_group, descending=True)
                all_losses.append(losses_per_group[sorted_indices])
            else:
                sorted_indices = np.arange(len(group))
            sliced_data_indices.append(group[sorted_indices])
            all_correct.append(correct_by_group[sorted_indices])
            all_probs.append(probs_by_group[sorted_indices])
    # Save GPU memory
    for p in net.parameters():
        p = p.detach().cpu() 
    net.to(torch.device('cpu')) 
    return sliced_data_indices, all_losses, all_correct, all_probs


def compute_pseudolabels(net, dataset, batch_size, args, criterion=None, 
                         dataloader=None):
    net.eval()
    if dataloader is None:
        new_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers)
    else:
        new_loader = dataloader
        dataset = dataloader.dataset
    all_outputs = []
    all_predicted = []
    all_correct = []
    all_correct_spurious = []
    all_losses = []
    net.to(args.device)

    with torch.no_grad():
        targets_s = dataset.targets_all['spurious']
        for batch_ix, data in enumerate(tqdm(new_loader)):
            inputs, labels, data_ix = data
            labels_spurious = torch.tensor(
                [targets_s[ix] for ix in data_ix]).to(args.device)

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = get_output(net, inputs, labels, args)
            _, predicted = torch.max(outputs.data, 1)
            all_outputs.append(outputs.detach().cpu())
            all_predicted.append(predicted.detach().cpu())
            if args.weigh_slice_samples_by_loss:
                assert criterion is not None, 'Need to specify criterion'
                loss = criterion(outputs, labels)
                all_losses.append(loss.detach().cpu())

            # Save correct
            correct = (predicted == labels).to(torch.device('cpu'))
            correct_spurious = (predicted == labels_spurious).to(torch.device('cpu'))
            all_correct.append(correct)
            all_correct_spurious.append(correct_spurious)
            
            inputs = inputs.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))
            outputs = outputs.to(torch.device('cpu'))
            predicted = predicted.to(torch.device('cpu'))

    pseudo_labels = torch.hstack(all_predicted)
    outputs = torch.vstack(all_outputs)
    correct = torch.hstack(all_correct)
    correct_spurious = torch.hstack(all_correct_spurious)
    if len(all_losses) > 0:
        all_losses = torch.hstack(all_losses)
    else:
        all_losses = None
    return pseudo_labels, outputs, correct, correct_spurious, all_losses


def train_spurious_model(train_loader, args, resample=False,
                         return_loaders=False, test_loader=None,
                         test_criterion=None):
    train_indices, train_indices_spurious = train_val_split(train_loader.dataset,
                                                            val_split=args.spurious_train_split, 
                                                            seed=args.seed)
    
    train_targets_all = train_loader.dataset.targets_all
    unique_target_counts = np.unique(train_targets_all['target'][train_indices_spurious],
                                     return_counts=True)
    print(f'Target values in spurious training data: {unique_target_counts}')
    
    train_set_new = get_resampled_set(train_loader.dataset,
                                      train_indices,
                                      copy_dataset=True)
    train_set_spurious = get_resampled_set(train_loader.dataset,
                                           train_indices_spurious,
                                           copy_dataset=True)

    train_loader_new = DataLoader(train_set_new,
                                  batch_size=args.bs_trn,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    train_loader_spurious = DataLoader(train_set_spurious,
                                       batch_size=args.bs_trn,
                                       shuffle=False,
                                       num_workers=args.num_workers)
    if resample is True:
        resampled_indices = get_resampled_indices(train_loader_spurious,
                                                  args,
                                                  args.resample_class)
        train_set_resampled = get_resampled_set(train_set_spurious,
                                                resampled_indices)
        train_loader_spurious = DataLoader(train_set_resampled,
                                           batch_size=args.bs_trn,
                                           shuffle=True,
                                           num_workers=args.num_workers)
        
    net = get_net(args)
    optim = get_optim(net, args, model_type='spurious')
    criterion = get_criterion(args)
    
    log_test_results = True if test_loader is not None else False

    outputs = train_model(net, optim, criterion,
                          train_loader=train_loader_spurious,
                          val_loader=train_loader_new,
                          args=args, epochs=args.max_epoch_s,
                          log_test_results=log_test_results,
                          test_loader=test_loader,
                          test_criterion=test_criterion)
    
    if return_loaders:
        return net, outputs, (train_loader_new, train_loader_spurious)
    return net, outputs, None


def train_batch_model(train_loader, sliced_data_indices, args,
                      val_loader, test_loader=None):
    """
    Train a single model with minibatch SGD aggregating and shuffling the sliced data indices - Updated with val loader
    """
    net = get_net(args, pretrained=False)
    optim = get_optim(net, args, model_type='pretrain')
    criterion = get_criterion(args)
    test_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    indices = np.hstack(sliced_data_indices)
    heading = f'Training on aggregated slices'
    print('-' * len(heading))
    print(heading)
    sliced_val_loader = val_loader
    sliced_train_sampler = SubsetRandomSampler(indices)
    sliced_train_loader = DataLoader(train_loader.dataset,
                                     batch_size=args.bs_trn,
                                     sampler=sliced_train_sampler,
                                     num_workers=args.num_workers)
    args.model_type = 'mb_slice'
    train_model(net, optim, criterion, sliced_train_loader,
                sliced_val_loader, args, 0, args.max_epoch,
                True, test_loader, test_criterion)
    return net
