"""
Training, evaluating, calculating embeddings functions
"""
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.slice_dataset import SliceDataset
from network import get_criterion, get_optim
from network import save_checkpoint, get_output
from utils import print_header
from utils.logging import summarize_acc
from utils.metrics import compute_roc_auc
from activations import compute_activation_mi, save_activations, compute_align_loss


def train_model(net, optimizer, criterion, train_loader, val_loader,
                args, start_epoch=0, epochs=None, log_test_results=False,
                test_loader=None, test_criterion=None,
                checkpoint_interval=None, scheduler=None):
    """
    Train model for specified number of epochs

    Args:
    - net (torch.nn.Module): Pytorch model network
    - optimizer (torch.optim): Model optimizer
    - criterion (torch.nn.Criterion): Pytorch loss function
    - train_loader (torch.utils.data.DataLoader): Training dataloader
    - val_loader (torch.utils.data.DataLoader): Validation dataloader
    - args (argparse): Experiment args
    - start_epoch (int): Which epoch to start from
    - epochs (int): Number of epochs to train
    - log_test_results (bool): If true evaluate model on test set after each epoch and save results
    - test_loader (torch.utils.data.DataLoader): Testing dataloader
    - test_criterion (torch.nn.Criterion): Pytorch testing loss function, most likely has reduction='none'
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    """
    try:
        if args.load_encoder is True or args.train_encoder is True:
            net.eval()
        else:
            net.train()
    except:
        net.train()
        
    # Test this?
    net.train()
    max_robust_test_acc = 0
    max_robust_epoch = None
    max_robust_test_group_acc = None
    all_acc = []
    
    
    epochs = args.max_epoch if epochs is None else epochs
    net.to(args.device)
    scheduler_ = scheduler if args.optim == 'AdamW' else None
    for epoch in range(start_epoch, start_epoch + epochs):
        train_outputs = train(net, train_loader, optimizer, criterion, args, scheduler_)
        running_loss, correct, total, correct_by_groups, total_by_groups = train_outputs
        
        if checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(net, optimizer, running_loss,
                            epoch, batch=0, args=args,
                            replace=True, retrain_epoch=None)
        
        val_outputs = evaluate(net, val_loader, criterion, args, testing=True)
        val_running_loss, val_correct, val_total, correct_by_groups_v, total_by_groups_v, correct_indices = val_outputs
        if (epoch + 1) % args.log_interval == 0:
            print(f'Epoch: {epoch + 1:3d} | Train Loss: {running_loss / total:<.3f} | Train Acc: {100 * correct / total:<.3f} | Val Loss: {val_running_loss / val_total:<.3f} | Val Acc: {100 * val_correct / val_total:<.3f}')
            
        if args.verbose is True:
            print('Training:')
            summarize_acc(correct_by_groups, total_by_groups)
            
        if args.verbose is True:
            print('Validating:')
            summarize_acc(correct_by_groups_v, total_by_groups_v)

        if args.optim == 'sgd' and scheduler is not None:
            group_acc = []
            for yix, y_group in enumerate(correct_by_groups_v):
                y_correct = []
                y_total = []
                for aix, a_group in enumerate(y_group):
                    if total_by_groups_v[yix][aix] > 0:
                        acc = a_group / total_by_groups_v[yix][aix]
                        # seed 1
                        if args.seed == 1:
                            if yix == aix:
                                y_correct.append(a_group)
                                y_total.append(total_by_groups_v[yix][aix])
                        else:
                            y_correct.append(a_group)
                            y_total.append(total_by_groups_v[yix][aix])
    #                         group_acc.append(acc)
                group_acc.append(np.sum(y_correct) /
                                 np.sum(y_total))
    #             for gix, group_correct in enumerate(correct_by_groups_v):
    #                 if total_by_groups_v[gix] > 0:
    #                     group_acc.append(group_correct / total_by_groups_v[gix])
            group_avg_acc = np.mean(group_acc)
            print(group_acc)
            print(group_avg_acc)
            scheduler.step(group_avg_acc)
            
        if log_test_results:
            assert test_loader is not None
            test_outputs = test_model(net, test_loader, test_criterion, args, epoch, mode='Training')
            test_running_loss, test_correct, test_total, correct_by_groups_t, total_by_groups_t, correct_indices, all_losses, losses_by_groups = test_outputs
            
            robust_test_acc = summarize_acc(correct_by_groups_t,
                                            total_by_groups_t)
            all_acc.append(robust_test_acc)
            if robust_test_acc >= max_robust_test_acc:
                max_robust_test_acc = robust_test_acc
                args.max_robust_acc = max_robust_test_acc
                max_robust_epoch = epoch
                max_robust_test_group_acc = (correct_by_groups_t,
                                             total_by_groups_t)
                
            plt.plot(all_acc)
            plt.title(f'Worst-group test accuracy (max acc: {args.max_robust_acc:<.4f})')
            figpath = os.path.join(args.results_path, f'ta-{args.experiment_name}.png')
            plt.savefig(figpath)
            plt.close()
            
            max_robust_metrics = (max_robust_test_acc, max_robust_epoch,
                                  max_robust_test_group_acc)
            if epoch + 1 == start_epoch + epochs:
                return net, max_robust_metrics, all_acc
            
    return (val_running_loss, val_correct, val_total, correct_by_groups, total_by_groups, correct_indices)


def test_model(net, test_loader, criterion, args, epoch, mode='Testing'):
    net.eval()
    test_running_loss, test_correct, test_total, correct_by_groups, total_by_groups, correct_indices, all_losses, losses_by_groups = evaluate(
        net, test_loader, criterion, args, testing=True, return_losses=True)
    acc_by_groups = correct_by_groups / total_by_groups
    if args.dataset != 'civilcomments':
        loss_header_1 = f'Avg Test Loss: {test_running_loss / test_total:<.3f} | Avg Test Acc: {100 * test_correct / test_total:<.3f}'
        loss_header_2 = f'Robust Loss: {np.max(losses_by_groups):<.3f} | Best Loss: {np.min(losses_by_groups):<.3f}'
        print_header(loss_header_1, style='top')
        print(loss_header_2)
    loss_header_3 = f'Robust Acc: {100 * np.min(acc_by_groups):<.3f} | Best Acc: {100 * np.max(acc_by_groups):<.3f}'
    # print_header(loss_header_1, style='top')
    # print_header(loss_header_2, style='bottom')
#     if args.verbose is True:
    
    print_header(loss_header_3, style='bottom')
    print(f'{mode}, Epoch {epoch}:')
    min_acc = summarize_acc(correct_by_groups, total_by_groups)
    
    if mode == 'Testing':
        # min_acc = summarize_acc(correct_by_groups, total_by_groups)
        if min_acc > args.max_robust_acc:
            max_robust_acc = min_acc  # Outsourced this
        else:
            max_robust_acc = args.max_robust_acc

        # Compute MI of activations
        attributes = ['target']
        if args.dataset != 'civilcomments':
            attributes.append('spurious')
        
        attribute_names = []
        
        embeddings, _ = save_activations(net, test_loader, args)
        mi_attributes = compute_activation_mi(attributes, test_loader, 
                                              method='logistic_regression',
                                              classifier_test_size=0.5,
                                              max_iter=5000,
                                              model=net,
                                              embeddings=embeddings, 
                                              seed=args.seed, args=args)
        for ix, attribute in enumerate(attributes):
            name = f'embedding_mutual_info_{attribute}'
            if name not in args.test_metrics:
                args.test_metrics[name] = []
            attribute_names.append(name)
            
        # Compute Loss Align
        # Only sensible now for waterbirds and ColoredMNIST
        if args.dataset in ['waterbirds', 'colored_mnist']:
            align_loss_metric_values = []
            align_loss_metrics = ['target', 'spurious']
            for align_loss_metric in align_loss_metrics:
                align_loss = compute_align_loss(embeddings, test_loader,
                                                measure_by=align_loss_metric,
                                                norm=True)
                align_loss_metric_values.append(align_loss)
                if f'loss_align_{align_loss_metric}' not in args.test_metrics:
                    args.test_metrics[f'loss_align_{align_loss_metric}'] = []

        for yix, y_group in enumerate(correct_by_groups):
            for aix, a_group in enumerate(y_group):
                args.test_metrics['epoch'].append(epoch + 1)
                args.test_metrics['target'].append(yix)  # (y_group)
                args.test_metrics['spurious'].append(aix)  # (a_group)
                args.test_metrics['acc'].append(acc_by_groups[yix][aix])
                try:
                    args.test_metrics['loss'].append(losses_by_groups[yix][aix])
                except:
                    args.test_metrics['loss'].append(-1)
                # Change this depending on setup
                args.test_metrics['model_type'].append(args.model_type)
                args.test_metrics['mutual_info_y_z'].append(args.mi_resampled)
                args.test_metrics['robust_acc'].append(min_acc)
                args.test_metrics['max_robust_acc'].append(max_robust_acc)

                # Mutual Info:
                for ix, name in enumerate(attribute_names):
                    args.test_metrics[name].append(mi_attributes[ix])
                    
                if args.dataset in ['waterbirds', 'colored_mnist']:
                    for alix, align_loss_metric in enumerate(align_loss_metrics):
                        args.test_metrics[f'loss_align_{align_loss_metric}'].append(align_loss_metric_values[alix])      
    else:
        summarize_acc(correct_by_groups, total_by_groups)
                
    return (test_running_loss, test_correct, test_total, correct_by_groups, total_by_groups, correct_indices, all_losses, losses_by_groups)


def train(net, dataloader, optimizer, criterion, args, scheduler=None):
    running_loss = 0.0
    correct = 0
    total = 0
    
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_t = dataloader.dataset.targets_all['target']

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)

    net.train()
    net.zero_grad()
    
    for i, data in enumerate(tqdm(dataloader)):
        inputs, labels, data_ix = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        # print(data_ix[0], data_ix[-1])
        labels_spurious = [targets_s[ix] for ix in data_ix]

        # Add this here to generalize NLP, CV models
        outputs = get_output(net, inputs, labels, args)
        loss = criterion(outputs, labels)
        
        if args.arch == 'bert-base-uncased_pt' and args.optim == 'AdamW':
            loss.backward()
            # Toggle this?
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.max_grad_norm)
            # Just keep args.max_grad_norm the same here^ regardless of spurious model?
            
            # In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
            # Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # optimizer.step()
            net.zero_grad()
        elif scheduler is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            net.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save performance
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        all_correct = (predicted == labels).detach().cpu()
        correct += all_correct.sum().item()
        running_loss += loss.item()
        
        # Save group-wise accuracy
        labels_target = labels.detach().cpu().numpy()
        for ix, s in enumerate(labels_spurious):
            y = labels_target[ix]
            correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
            total_by_groups[int(y)][int(s)] += 1

        # Clear memory
        inputs = inputs.to(torch.device('cpu'))
        labels = labels.to(torch.device('cpu'))  
        outputs = outputs.to(torch.device('cpu'))
        loss = loss.to(torch.device('cpu'))
        del outputs; del inputs; del labels; del loss
        
    return running_loss, correct, total, correct_by_groups, total_by_groups



def evaluate_civilcomments(net, dataloader, criterion, args):
    # Test this first
    dataset = dataloader.dataset
    metadata = dataset.metadata_array
    correct_by_groups = np.zeros([2, len(dataset._identity_vars)])
    total_by_groups = np.zeros(correct_by_groups.shape)
    
    # Bad way
    identity_to_ix = {}
    for idx, identity in enumerate(dataset._identity_vars):
        identity_to_ix[identity] = idx
    
    
    
    for identity_var, eval_grouper in zip(dataset._identity_vars, 
                                          dataset._eval_groupers):
        group_idx = eval_grouper.metadata_to_group(metadata).numpy()
        
        g_list, g_counts = np.unique(group_idx, return_counts=True)
        print(identity_var, identity_to_ix[identity_var])
        print(g_counts)
        
        for g_ix, g in enumerate(g_list):
            g_count = g_counts[g_ix]
        # g, g_counts in np.unique(group_idx, return_counts=True):
            # Only pick from positive identities
            # e.g. only 1 and 3 from here:
            #   0 y:0_male:0
            #   1 y:0_male:1
            #   2 y:1_male:0
            #   3 y:1_male:1
            n_total = g_counts[g_ix]  #  + g_counts[3]
            if g in [1, 3]:
                # n_correct = all_correct[group_idx == g].sum()
                class_ix = 0 if g == 1 else 1  # 1 y:0_male:1
                # correct_by_groups[class_ix][identity_var] += n_correct
                # total_by_groups[class_ix][identity_var] += n_total
                print(g_ix, g, n_total)
    
    
    net.to(args.device)
    net.eval()  # Commented this in, does it change things? i.e. batchnorm
    total_correct = 0
    with torch.no_grad():
        all_predictions = []
        all_correct = []
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # Add this here to generalize NLP, CV models
            outputs = get_output(net, inputs, labels, args)
            # loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).detach().cpu()
            total_correct += correct.sum().item()
            all_correct.append(correct)
            all_predictions.append(predicted.detach().cpu())
            
            
            inputs = inputs.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))
            outputs = outputs.to(torch.device('cpu'))
            del inputs; del labels; del outputs
        
        all_correct = torch.cat(all_correct).numpy()
        all_predictions = torch.cat(all_predictions)
    
    # Evaluate predictions
    dataset = dataloader.dataset
    y_pred = all_predictions  # torch.tensors
    y_true = dataset.y_array
    metadata = dataset.metadata_array
    
    correct_by_groups = np.zeros([2, len(dataset._identity_vars)])
    total_by_groups = np.zeros(correct_by_groups.shape)
    
    for identity_var, eval_grouper in zip(dataset._identity_vars, 
                                          dataset._eval_groupers):
        group_idx = eval_grouper.metadata_to_group(metadata).numpy()
        
        g_list, g_counts = np.unique(group_idx, return_counts=True)
        print(g_counts)
        
        idx = identity_to_ix[identity_var]
        
        for g_ix, g in enumerate(g_list):
            g_count = g_counts[g_ix]
        # g, g_counts in np.unique(group_idx, return_counts=True):
            # Only pick from positive identities
            # e.g. only 1 and 3 from here:
            #   0 y:0_male:0
            #   1 y:0_male:1
            #   2 y:1_male:0
            #   3 y:1_male:1
            n_total = g_count  # s[1] + g_counts[3]
            if g in [1, 3]:
                n_correct = all_correct[group_idx == g].sum()
                class_ix = 0 if g == 1 else 1  # 1 y:0_male:1
                correct_by_groups[class_ix][idx] += n_correct
                total_by_groups[class_ix][idx] += n_total
                
    # return running_loss, correct, total, correct_by_groups, total_by_groups, correct_indices, all_losses, losses_by_groups
    return 0, total_correct, len(dataset), correct_by_groups, total_by_groups, None, None, None


def evaluate(net, dataloader, criterion, args, testing=False, return_losses=False):
    if args.dataset == 'civilcomments':
        return evaluate_civilcomments(net, dataloader, criterion, args)
    
    
    # Validation
    running_loss = 0.0
    all_losses = []
    correct = 0
    total = 0

    targets_s = dataloader.dataset.targets_all['spurious'].astype(int)
    targets_t = dataloader.dataset.targets_all['target'].astype(int)

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    auroc_by_groups = np.zeros([len(np.unique(targets_t)),
                                len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)
    
#     if args.compute_auroc is True:
#         correct_by_groups = [[[] for _ in range(len(np.unique(targets_t)))]
#                              for _ in range(len(np.unique(targets_s)))]
#         total_by_groups = [[[] for _ in range(len(np.unique(targets_t)))]
#                            for _ in range(len(np.unique(targets_s)))]

    correct_indices = []
    net.to(args.device)
    net.eval()  # Commented this in, does it change things? i.e. batchnorm

    with torch.no_grad():
        all_probs = []
        all_targets = []
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            labels_spurious = [targets_s[ix] for ix in data_ix]

            # Add this here to generalize NLP, CV models
            outputs = get_output(net, inputs, labels, args)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_correct = (predicted == labels).detach().cpu()
            correct += all_correct.sum().item()
            loss_r = loss.mean() if return_losses else loss
            running_loss += loss_r.item()
            all_losses.append(loss.detach().cpu().numpy())
            
            # For AUROC
            if args.compute_auroc is True:
                print(labels)
                print(F.softmax(outputs, dim=1).detach().cpu()[:, 1])
                print((F.softmax(outputs, dim=1).detach().cpu()[:, 1]).shape)
                all_probs.append(F.softmax(outputs, dim=1).detach().cpu()[:, 1])  # For AUROC
                all_targets.append(labels.detach().cpu())

            correct_indices.append(all_correct.numpy())

            if testing:
                for ix, s in enumerate(labels_spurious):
                    y = labels.detach().cpu().numpy()[ix]
#                     if args.compute_auroc is True:
#                         # Slightly abusing this naming
#                         correct_by_groups[int(y)][int(s)].append(
#                             outputs[ix][0].item())  # for binary
#                         total_by_groups[int(y)][int(s)].append(
#                             labels[ix].item())
#                     else:
                    correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
                    total_by_groups[int(y)][int(s)] += 1
                    if return_losses:
                        losses_by_groups[int(y)][int(s)] += loss[ix].item()
            inputs = inputs.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))
            outputs = outputs.to(torch.device('cpu'))
            loss = loss.to(torch.device('cpu'))
            loss_r = loss_r.to(torch.device('cpu'))
            del inputs; del labels; del outputs
            
        if args.compute_auroc is True:
            targets_cat, probs_cat = torch.cat(all_targets), torch.cat(all_probs)
            auroc = compute_roc_auc(targets_cat, probs_cat)
            
            malignant_indices = np.where(targets_t == 1)[0]
            for i in range(len(auroc_by_groups[1])):
                auroc_by_groups[1][i] = auroc
            
            benign_indices = np.where(targets_t == 0)[0]
            for s in np.unique(targets_s[benign_indices]):
#             for s in [0, 1]:
                spurious_indices = np.where(targets_s[benign_indices] == s)[0]
                paired_auroc_indices = np.union1d(malignant_indices,
                                                  benign_indices[spurious_indices])
                auroc = compute_roc_auc(targets_cat[paired_auroc_indices],
                                        probs_cat[paired_auroc_indices])
                auroc_by_groups[0][s] = auroc
                
            # Hack
            args.auroc_by_groups = auroc_by_groups
            min_auroc = np.min(args.auroc_by_groups.flatten())
            print('-' * 18)
            print(f'AUROC by group:')
            for yix, y_group in enumerate(auroc_by_groups):
                for aix, a_group in enumerate(y_group):
                    print(f'{yix}, {aix}  auroc: {auroc_by_groups[yix][aix]:>5.3f}')
            try:
                if min_auroc > args.robust_auroc:
                    print(f'- New max robust AUROC: {min_auroc:<.3f}')
                    args.robust_auroc = min_auroc
            except:
                print(f'- New max robust AUROC: {min_auroc:<.3f}')
                args.robust_auroc = min_auroc
                
            
#             for i in range(len(correct_by_groups)):
#                 for j in range(len(correct_by_groups[i])):
#                     if len(correct_by_groups[i][j]) > 0:
#                         correct_by_groups[i][j] = np.array(
#                             correct_by_groups[i][j])
#                         total_by_groups[i][j] = np.array(
#                             total_by_groups[i][j])
#                         correct_by_groups[i][j] = roc_auc_score(
#                             total_by_groups[i][j], correct_by_groups[i][j])
#                     else:
#                         correct_by_groups[i][j] = auroc  # Just say 1
#                     total_by_groups[i][j] = 1
    if testing:
        if return_losses:
            all_losses = np.concatenate(all_losses)
            return running_loss, correct, total, correct_by_groups, total_by_groups, correct_indices, all_losses, losses_by_groups
        return running_loss, correct, total, correct_by_groups, total_by_groups, correct_indices
    return running_loss, correct, total, correct_indices


def pretrain_model(net, optimizer, criterion, dataset_class,
                   data_generation_fn, data_args, args):
    train_data_ = dataset_class(data_generation_fn, data_args, train=True)
    test_data = dataset_class(data_generation_fn, data_args, train=False)

    train_set_len = train_data_.__len__()
    val_ix = int(np.round(args.val_split * train_set_len))
    # Shuffle train and val splits
    all_indices = np.arange(train_set_len)
    np.random.seed(args.seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[val_ix:]
    val_indices = all_indices[:val_ix]

    train_data = SliceDataset(train_data_, indices=train_indices,
                              subsample_labels=False, subsample_groups=False)
    val_data = SliceDataset(train_data_, indices=val_indices,
                            subsample_labels=False, subsample_groups=False)
    # train_data = train_data_
    # val_data = dataset_class(data_generation_fn, data_args, train=False)

    train_loader = DataLoader(train_data, batch_size=args.bs_trn,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.bs_trn,
                            shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.bs_trn,
                             shuffle=False, num_workers=args.num_workers)

    net.to(args.device)
    for epoch in range(args.max_epoch):
        running_loss, correct, total, _, _ = train(net, train_loader,
                                             optimizer, criterion, args)
        val_running_loss, val_correct, val_total, _ = evaluate(net, val_loader,
                                                               criterion, args)
        if (epoch + 1) % args.log_interval == 0:
            print(f'Epoch: {epoch + 1:3d} | Train Loss: {running_loss / total:<.3f} | Train Acc: {100 * correct / total:<.3f} | Val Loss: {val_running_loss / val_total:<.3f} | Val Acc: {100 * val_correct / val_total:<.3f}')
    test_running_loss, test_correct, test_total, correct_by_groups, total_by_groups, _ = evaluate(
        net, test_loader, criterion, args, testing=True)
    print(f'Epoch: {epoch + 1:3d} | Test Loss: {test_running_loss / test_total:<.3f} | Test Acc: {100 * test_correct / test_total:<.3f}')
    summarize_acc(correct_by_groups, total_by_groups)

    return net, (train_data, val_data, test_data), (train_loader, val_loader, test_loader)
