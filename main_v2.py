"""
All functions for training CNC again, but this time using batch-wise comparisons only

Special parameters to pay attention to:
* Whether inferred groups should be balanced (resampling: oversampling or undersampling?)


# Sample command
python main_v2.py --dataset waterbirds --arch resnet50_pt --pretrained_spurious_path "./model/waterbirds/waterbirds_erm_regularized.pt" --resample_by_group upsample --bs_trn 128 --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 --weight_decay 1e-3 --weight_decay_c 1e-3 --temperature 0.1 --max_epoch 300 --no_projection_head --contrastive_weight 0.75 --verbose --replicate 42 --seed 42

python main_v2.py --dataset waterbirds --arch resnet50_pt --pretrained_spurious_path "./model/waterbirds/waterbirds_erm_regularized.pt" --resample_by_group upsample --bs_trn 128 --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 --weight_decay 1e-3 --weight_decay_c 1e-3 --temperature 0.1 --max_epoch 300 --no_projection_head --contrastive_weight 0.75 --verbose --replicate 42 --seed 42 --slice_with true

python main_v2.py --dataset waterbirds --arch resnet50_pt --resample_by_group upsample --bs_trn 128 --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 --weight_decay 1e-3 --weight_decay_c 1e-3 --temperature 0.1 --max_epoch 300 --no_projection_head --contrastive_weight 0.75 --verbose --max_epoch_s 5 --bs_trn_s 128 --lr_s 1e-3 --weight_decay_s 1e-4 --replicate 42 --seed 42 
"""
import os
import sys
import copy
import argparse
import importlib

from os.path import join, exists
from numpy.lib.function_base import corrcoef

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

from utils import print_header, set_seed, free_gpu
from utils.logging import Logger, log_args, summarize_acc, log_data
from datasets import initialize_data, train_val_split, get_resampled_set

# Model
from network import get_net, get_optim, get_criterion, load_pretrained_model, save_checkpoint
from network import CNN, MLP
# from contrastive_network import ContrastiveNet
from resnet import *


def init_save_paths(args):
    # Update saving paths
    new_model_path = join(args.model_path, args.dataset)
    new_image_path = join(args.image_path, args.dataset)
    new_log_path = join(args.log_path, args.dataset)
    new_results_path = join(args.results_path, args.dataset)
    if not exists(new_model_path):
        os.makedirs(new_model_path)
    if not exists(new_image_path):
        os.makedirs(new_image_path)
    if not exists(new_log_path):
        os.makedirs(new_log_path)
    if not exists(new_results_path):
        os.makedirs(new_results_path)
    # Make more granular - save specific folders per experiment configs
    new_model_path = join(new_model_path, args.experiment_configs)
    new_image_path = join(new_image_path, args.experiment_configs)
    new_log_path = join(new_log_path, args.experiment_configs)
    new_results_path = join(new_results_path, args.experiment_configs)
    if not exists(new_model_path):
        os.makedirs(new_model_path)
    if not exists(new_image_path):
        os.makedirs(new_image_path)
    if not exists(new_log_path):
        os.makedirs(new_log_path)
    if not exists(new_results_path):
        os.makedirs(new_results_path)
    args.model_path = new_model_path
    args.image_path = new_image_path
    args.log_path = new_log_path
    args.results_path = new_results_path

    # Only save UMAPs of model representations?
    args.image_path = os.path.join(args.image_path, 'contrastive_umaps')
    if not os.path.exists(args.image_path):
        os.makedirs(args.image_path)


def init_experiment(args):
    """
    Initialize experiment, save name, seeds
    """
    args.criterion = 'cross_entropy'
    args.pretrained = False

    # BERT Defaults
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    # Keep these the same for the spurious model
    args.max_grad_norm_s = 1.0
    args.adam_epsilon_s = 1e-8
    args.warmup_steps_s = 0
    # And the same for grad-aligned finetuning
    args.grad_max_grad_norm = 1.0
    args.grad_adam_epsilon = 1e-8
    args.grad_warmup_steps = 0

    args.device = torch.device('cuda:0') if torch.cuda.is_available(
    ) and not args.no_cuda else torch.device('cpu')

    # Visualizations
    args.img_file_type = 'png'
    args.display_image = False
    args.image_path = './images'

    # Misc. - can't spell
    args.log_interval = 1
    args.log_path = './logs'
    args.results_path = './results'
    args.model_path = './model'
    args.image_path = './images'
    args.img_file_type = '.png'

    # Slicing
    args.loss_factor = 1
    args.supersample_labels = False
    args.subsample_labels = False
    args.weigh_slice_samples_by_loss = True  # just to compute losses

    # Legacy args here
    args.val_split = 0.1
    args.spurious_train_split = 0.2
    args.subsample_groups = False
    args.train_method = 'sc'  # Because "slicing" by U-MAP, retrain

    # Setup experiment configs for save paths
    if args.dataset in ['waterbirds', 'waterbirds_r', 'cxr', 'multinli']:  # 'celebA'
        experiment_configs = f'config-tn={args.target_name}-cn={args.confounder_names}'
    elif args.dataset == 'colored_mnist':
        if args.p_corr_by_class is None:
            p_corr_arg = args.p_correlation
        else:
            p_corr_arg = '_'.join([str(pcc[0])
                                  for pcc in args.train_class_ratios])

        train_classes_arg = '_'.join([str(tc) for tc in args.train_classes])
        experiment_configs = f'config-p={p_corr_arg}-cmap={args.data_cmap}-test={args.test_shift}{test_cmap}{flipped}-tr_c={train_classes_arg}'

        if args.train_class_ratios is not None:
            tcr = '_'.join([str(tcr[0]) for tcr in args.train_class_ratios])
            experiment_configs += f'-tr_cr={tcr}'
    else:
        experiment_configs = f'config'
    args.experiment_configs = experiment_configs

    # Set up saving paths
    init_save_paths(args)

    # Set up experiment name
    model_params = f'nph={int(args.no_projection_head)}-pd={args.projection_dim}-bf={args.batch_factor}-t={args.temperature}-sp={int(args.single_pos)}-cw={args.contrastive_weight}-ma={int(args.majority_anchor)}'
    model_params += f'-me={args.max_epoch}-bst={args.bs_trn}-o={args.optim}-lr={args.lr}-mo={args.momentum}-wd={args.weight_decay}'
    model_params_s = f'spur-me={args.max_epoch_s}-bst={args.bs_trn_s}-lr={args.lr_s}-mo={args.momentum_s}-wd={args.weight_decay_s}-sts={args.spurious_train_split}'

    args.experiment_name = f'cnc_v2'
    if args.dataset == 'colored_mnist':
        args.experiment_name += f'-cmnist_p{args.p_correlation}-bs_trn_s={args.bs_trn_s}'
    else:
        args.experiment_name += f'-{args.dataset}'
    if args.no_projection_head:
        args.experiment_name += f'-nph'

    resample_by_group = 0 if args.resample_by_group == '' else args.resample_by_group[0]
    args.experiment_name += f'-sw={args.slice_with[:2]}-rs={resample_by_group}'
    args.experiment_name += f'-{model_params}-{model_params_s}-s={args.seed}-r={args.replicate}'
    print(f'Experiment name: {args.experiment_name}')


def init_logging(args):
    if os.path.exists(args.log_path) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'
    logger = Logger(os.path.join(args.log_path,
                                 f'log-{args.experiment_name}.txt'), mode)
    log_args(args, logger)
    sys.stdout = logger
    args.resume = resume


def init_args():
    parser = argparse.ArgumentParser(description='Correct-n-Contrast')

    # Model
    parser.add_argument('--arch', choices=['mlp', 'cnn', 'resnet50', 'resnet50_pt'],
                        required=True)
    parser.add_argument('--bs_trn', type=int, default=128)
    parser.add_argument('--bs_val', type=int, default=128)

    # Data
    parser.add_argument('--dataset', type=str,
                        choices=['waterbirds', 'cmnist', 'celebA', 'civilcomments'])
    # Initial slicing for anchor-positive-negative generation
    parser.add_argument('--slice_with', type=str, default='rep',
                        choices=['rep', 'pred', 'pred_and_rep', 'true'])
    parser.add_argument('--rep_cluster_method', type=str,
                        default='gmm', choices=['kmeans', 'gmm'])
    parser.add_argument('--resample_by_group', type=str, default='',
                        choices=['', 'upsample', 'subsample'])

    # Training
    # Contrastive model
    parser.add_argument('--train_encoder', default=False, action='store_true')
    parser.add_argument('--no_projection_head',
                        default=False, action='store_true')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--batch_factor', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--single_pos', default=False, action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--majority_anchor',
                        default=False, action='store_true')
    # General training hyperparameters
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'AdamW'])  # Keep the same for all stages
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_c', type=float, default=-1)
    # Stage 1 "spurious" model
    parser.add_argument('--pretrained_spurious_path', default='', type=str)
    parser.add_argument('--max_epoch_s', type=int, default=1)
    parser.add_argument('--optim_s', type=str, default='sgd',
                        choices=['sgd', 'adam', 'AdamW'])
    parser.add_argument('--bs_trn_s', type=int, default=32)
    parser.add_argument('--lr_s', type=float, default=1e-3)
    parser.add_argument('--momentum_s', type=float, default=0.9)
    parser.add_argument('--weight_decay_s', type=float, default=5e-4,)
    parser.add_argument('--slice_temp', type=float, default=10)

    # Logging
    parser.add_argument('--log_loss_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--grad_checkpoint_interval', type=int, default=50)
    parser.add_argument('--log_visual_interval', type=int, default=100)
    parser.add_argument('--log_grad_visual_interval', type=int, default=50)
    parser.add_argument('--verbose', default=False, action='store_true')

    # Additional
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reproduce', default=False, action='store_true')
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--new_slice', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--evaluate', default=False, action='store_true')

    # Colored MNIST specific
    # - Ignored if args.dataset != 'colored_mnist'
    parser.add_argument('--data_cmap', type=str, default='hsv',
                        help="Color map for digits. If solid, color all digits the same color")
    parser.add_argument('--test_cmap', type=str, default='',
                        help="Color map for digits. Solid colors applies same color to all digits. Only applies if specified, and automatically changes test_shift to 'generalize'")
    parser.add_argument('-pc', '--p_correlation', type=float, default=0.9,
                        help="Ratio of majority group size to total size")
    parser.add_argument('-pcc', '--p_corr_by_class', type=float, nargs='+', action='append',
                        help="If specified, p_corr for each group, e.g. -pcc 0.9 -pcc 0.9 -pcc 0.9 -pcc 0.9 -pcc 0.9 is the same as -pc 0.9")
    parser.add_argument('-tc', '--train_classes', type=int, nargs='+', action='append',
                        help="How to set up the classification problem, e.g. -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9")
    parser.add_argument('-tcr', '--train_class_ratios', type=float, nargs='+', action='append',
                        help="If specified, introduce class imbalance by only including the specified ratio of datapoints per class, e.g. for original ratios: -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 ")
    parser.add_argument('--test_shift', type=str, default='random',
                        help="How to shift the colors encountered in the test set - choices=['random', 'unseen', 'iid', 'shift_n' 'generalize']")
    parser.add_argument('--flipped', default=False, action='store_true',
                        help="If true, color background and leave digit white")
    args = parser.parse_args()
    return args


# ---------------------------------
# Training and Evaluation Functions
# ---------------------------------
def train(model, train_loader, val_loader, optimizer, criterion, epochs, args,
          test_loader=None, scheduler=None):
    best_model_state_dict = copy.deepcopy(model.state_dict())
    max_robust_val_acc = 0
    max_robust_val_epoch = None

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f'Epoch {epoch}')
        else:
            pbar.set_description(
                f'Epoch {epoch} | Avg val acc: {val_avg_acc:.3f}% | Robust val acc: {val_robust_acc:.3f}%')

        train_outputs = train_epoch(model, train_loader, optimizer,
                                    criterion, args, scheduler)
        running_loss, correct, total, correct_by_groups, total_by_groups = train_outputs
        if args.verbose:
            _, _ = summarize_acc(correct_by_groups, total_by_groups)
        val_outputs = evaluate_epoch(model, val_loader, criterion, args)
        val_running_loss, val_correct, val_total, correct_by_groups_v, total_by_groups_v = val_outputs
        val_avg_acc = val_correct / val_total * 100
        _, val_robust_acc = summarize_acc(correct_by_groups_v,
                                          total_by_groups_v,
                                          stdout=args.verbose)
        if val_robust_acc > max_robust_val_acc:
            max_robust_val_acc = val_robust_acc
            max_robust_val_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

        if test_loader is not None:
            test_outputs = evaluate_epoch(model, test_loader, criterion, args)
            test_running_loss, test_correct, test_total, correct_by_groups_t, total_by_groups_t = val_outputs
            test_avg_acc = test_correct / test_total * 100
            if args.verbose:
                _, test_robust_acc = summarize_acc(correct_by_groups_t,
                                                   total_by_groups_t,
                                                   stdout=True)
    return best_model_state_dict


def train_epoch(model, dataloader, optimizer, criterion, args, scheduler=None):
    return run_epoch(model, dataloader, optimizer, criterion, args,
                     train=True, scheduler=scheduler)


def evaluate_epoch(model, dataloader, criterion, args):
    return run_epoch(model, dataloader, optimizer=None, criterion=criterion, args=args,
                     train=False, scheduler=None)

#  Returns running_loss, correct, total, correct_by_groups, total_by_groups


def run_epoch(model, dataloader, optimizer, criterion, args, train, scheduler=None):
    running_loss = 0.0
    correct = 0
    total = 0

    targets_s = dataloader.dataset.targets_all['spurious']
    targets_t = dataloader.dataset.targets_all['target']

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)

    model.to(args.device)

    if train is True:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    model.zero_grad()

    pbar = tqdm(enumerate(dataloader))
    with torch.no_grad() if train is False else torch.enable_grad():
        for batch_ix, data in pbar:
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            labels_spurious = [targets_s[ix] for ix in data_ix]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            data_split = 'Eval'
            if train is True:
                data_split = 'Train'
                if args.arch == 'bert-base-uncased_pt' and args.optim == 'AdamW':
                    loss.backward()
                    # Toggle this?
                    if args.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    # optimizer.step()
                    model.zero_grad()
                elif scheduler is not None:
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    model.zero_grad()
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

            pbar.set_description(
                f'{data_split} | Batch Ix: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_ix, len(dataloader), running_loss / (batch_ix + 1),
                 100. * correct / total, correct, total)
            )

            # Clear memory
            inputs = inputs.cpu()
            labels = labels.cpu()
            outputs = outputs.cpu()
            loss = loss.cpu()
            del outputs
            del inputs
            del labels
            del loss

    return running_loss, correct, total, correct_by_groups, total_by_groups


# -----------------
# Stage 1 Functions
# -----------------
def train_stage_1_model(train_loader, args):
    train_indices, train_indices_stage_1 = train_val_split(train_loader.dataset,
                                                           val_split=args.spurious_train_split,
                                                           seed=args.seed)
    stage_1_val_set = get_resampled_set(train_loader.dataset,
                                        train_indices, copy_dataset=True)
    stage_1_train_set = get_resampled_set(train_loader.dataset,
                                          train_indices_stage_1,
                                          copy_dataset=True)
    stage_1_train_loader = DataLoader(stage_1_train_set, batch_size=args.bs_trn,
                                      shuffle=False, num_workers=args.num_workers)
    stage_1_val_loader = DataLoader(stage_1_val_set, batch_size=args.bs_val,
                                    shuffle=False, num_workers=args.num_workers)

    model = get_net(args)
    optim = get_optim(model, args, model_type='spurious')
    criterion = get_criterion(args)

    best_stage_one_model_state_dict = train(model,
                                            train_loader=stage_1_train_loader,
                                            val_loader=stage_1_val_loader,
                                            optimizer=optim,
                                            criterion=criterion,
                                            epochs=args.max_epoch_s,
                                            args=args)
    return model, best_stage_one_model_state_dict


def get_stage_1_predictions(train_loader, model, args):
    if args.slice_with == 'true':
        all_predictions = train_loader.dataset.targets_all['spurious']
        return all_predictions
    # Else:
    model.eval()
    model.to(args.device)
    all_predictions = []
    with torch.no_grad():
        for batch_ix, data in enumerate(tqdm(train_loader)):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted.cpu().numpy())

            outputs = outputs.cpu()
            predicted = predicted.cpu()
            inputs = inputs.cpu()
            labels = labels.cpu()
            del outputs
            del predicted
            del inputs
            del labels
    model.cpu()
    assert len(np.concatenate(all_predictions)) == len(
        train_loader.dataset.targets)
    return np.concatenate(all_predictions)


def get_group_resampled_loader(dataloader, predictions, args):
    if args.slice_with == 'true':
        predictions = dataloader.dataset.targets_all['spurious']
    np.random.seed(args.seed)
    data_indices = []
    for label in np.unique(predictions):
        group = np.where(predictions == label)[0]
        target_values = dataloader.dataset.targets[group]
        group_vals = np.unique(dataloader.dataset.targets[group],
                               return_counts=True)[1]
        sample_size = (np.min(group_vals) if args.resample_by_group == 'subsample' is True
                       else np.max(group_vals))
        sampled_indices = []
        for v in np.unique(target_values):
            group_indices = np.where(target_values == v)[0]
            if args.resample_by_group == 'subsample':
                sampling_size = np.min([len(group_indices), sample_size])
                replace = False
                p = None
            elif args.resample_by_group == 'upsample':
                sampling_size = np.max(
                    [0, sample_size - len(group_indices)])
                sampled_indices.append(group_indices)
                replace = True
                p = None
            sampled_indices.append(np.random.choice(
                group_indices, size=sampling_size, replace=replace, p=p))
        sampled_indices = np.concatenate(sampled_indices)
        data_indices.append(group[sampled_indices])
    data_indices = np.concatenate(data_indices)
    np.random.shuffle(data_indices)
    dataset = get_resampled_set(dataloader.dataset,
                                resampled_set_indices=data_indices,
                                copy_dataset=True)
    dataloader = DataLoader(dataset, batch_size=args.bs_trn,
                            shuffle=False, num_workers=args.num_workers)
    return dataloader


# -----------------------------
# Stage 2 Classes and Functions
# -----------------------------
class JointContrastiveNet(nn.Module):
    """
    Contrastive network where classifier and encoder layers are jointly optimized
    """

    def __init__(self, base_model_name, out_dim, projection_head=False,
                 task=None, num_classes=None, checkpoint=None):
        super(JointContrastiveNet, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        if base_model_name[-3:] == '_pt':
            self.pretrained = True
            base_model_name = base_model_name[:-3]
        else:
            self.pretrained = False
        print(f'Loading with {base_model_name} backbone')
        self.base_model_name = base_model_name
        self.encoder = self.init_basemodel(self.base_model_name)
        self.projection_head = projection_head
        self.encoder = self.init_projection_head(self.encoder,
                                                 out_dim,
                                                 project=projection_head)

    def init_basemodel(self, model_name):
        try:
            if 'resnet50' in model_name:
                model = resnet50(pretrained=self.pretrained)
                d = model.fc.in_features
                model.fc = nn.Linear(d, self.num_classes)
                self.activation_layer = 'backbone.avgpool'

            elif 'cnn' in model_name:
                model = CNN(num_classes=self.num_classes)
                self.activation_layer = torch.nn.ReLU

            elif 'mlp' in model_name:
                model = MLP(num_classes=self.num_classes,
                            hidden_dim=256)
                self.activation_layer = torch.nn.ReLU

            elif 'bert' in model_name:  # model_name = 'bert-base-uncased'
                raise NotImplementedError
                assert self.num_classes is not None
                assert self.task is not None
                config_class = BertConfig
                model_class = BertForSequenceClassification
                self.config = config_class.from_pretrained(model_name,
                                                           num_labels=self.num_classes,
                                                           finetuning_task=self.task)
                model = model_class.from_pretrained(model_name,
                                                    from_tf=False,
                                                    config=self.config)
                self.activation_layer = 'backbone.bert.pooler.activation'

            if self.checkpoint is not None:
                try:
                    state_dict = self.checkpoint['model_state_dict']
                    for k in list(state_dict.keys()):
                        if k.startswith('fc.') and 'bert' in model_name:
                            state_dict[f'classifier.{k[3:]}'] = state_dict[k]
                            del state_dict[k]

                    model.load_state_dict(state_dict)
                    print(f'Checkpoint loaded!')
                except Exception as e:
                    print(f'Checkpoint not loaded:')
                    print(f'- {e}')
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def init_projection_head(self, base_model, out_dim, project=True):
        if 'resnet' in self.base_model_name or 'cnn' in self.base_model_name or 'mlp' in self.base_model_name:
            dim_mlp = base_model.fc.in_features

            # self.classifier = nn.Linear(dim_mlp, self.num_classes)
            self.classifier = copy.deepcopy(base_model.fc)
            base_model.fc = nn.Identity(dim_mlp, -1)
            if project:
                # Add projection head
                self.projection_head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                     nn.ReLU(),
                                                     nn.Linear(dim_mlp, out_dim))
            else:
                self.projection_head = nn.Identity(dim_mlp, -1)

        elif 'bert' in self.base_model_name:
            dim_mlp = base_model.classifier.in_features

            self.classifier = copy.deepcopy(base_model.classifier)
            # print(self.classifier)
            if project:
                base_model.classifier = nn.Linear(dim_mlp, out_dim)
                base_model.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                      nn.ReLU(),
                                                      base_model.classifier)
            else:
                base_model.classifier = nn.Identity(dim_mlp, -1)
                # print(base_model.classifier)
        self.dim_mlp = dim_mlp
        return base_model

    def encode_and_classify(self, x):
        """
        Return outputs for both contrastive and classifier losses
        """
        if self.base_model_name == 'bert-base-uncased':
            input_ids, input_masks, segment_ids, labels = x
            outputs = self.encoder(input_ids=input_ids,
                                   attention_mask=input_masks,
                                   token_type_ids=segment_ids,
                                   labels=labels)
            if labels is None:
                return outputs.logits
            return outputs[1]  # [1] returns logits

        z = self.encoder(x)
        y = self.classifier(z)
        if self.projection_head is True:
            z = self.projection_head(x)
        return z, y

    def forward(self, x):
        if self.base_model_name == 'bert-base-uncased':
            input_ids, input_masks, segment_ids, labels = x
            outputs = self.encoder(input_ids=input_ids,
                                   attention_mask=input_masks,
                                   token_type_ids=segment_ids,
                                   labels=labels)
            if labels is None:
                return outputs.logits
            return outputs[1]  # [1] returns logits

        z = self.encoder(x)
        y = self.classifier(z)
        return y


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.temperature = args.temperature
        self.single_pos = args.single_pos
        self.sim = nn.CosineSimilarity(dim=1)
        self.majority_anchor = args.majority_anchor

    def forward(self, features, data_ix, targets_t, targets_p):
        targets_t = targets_t[data_ix]
        targets_p = targets_p[data_ix]
        loss = 0
        for aix, anchor in enumerate(features):
            # Get positives
            # print(targets_p.shape)
            # print(targets_p[aix].shape)
            # print(np.where(targets_p != targets_p[aix]))
            # print(f'anchor.shape: {anchor.shape}')
            pos_ix = np.where(np.logical_and(
                targets_t == targets_t[aix],
                targets_p != targets_p[aix]
            ))[0]
            positives = features[pos_ix]
            # print(f'positives.shape: {positives.shape}')
            # Get negatives
            neg_ix = np.where(np.logical_and(
                targets_t != targets_t[aix],
                targets_p == targets_p[aix]
            ))[0]
            negatives = features[neg_ix]

            if len(positives) > 0 and len(negatives) > 0:
                # print(f'negatives.shape: {negatives.shape}')
                exp_pos = self.compute_exp_sim(
                    anchor, positives, return_sum=False)
                exp_neg = self.compute_exp_sim(
                    anchor, negatives, return_sum=True)
                if self.single_pos:
                    log_probs = torch.log(exp_pos) - \
                        torch.log(exp_neg + exp_pos)
                else:
                    log_probs = (torch.log(exp_pos) -
                                 torch.log(exp_neg + exp_pos.sum(0, keepdim=True)))
                loss -= log_probs.mean()
                del exp_pos
                del exp_neg
                del log_probs
        return loss / len(features)

    def compute_exp_sim(self, anchor, pos_or_neg, return_sum=True):
        """
        Assume pos_or_neg := [pos_1, ... pos_N] or [neg_1, ..., neg_N]
        """
        sim = self.sim(anchor.view(1, -1), pos_or_neg)
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        if return_sum:
            exp_sim = exp_sim.sum(0, keepdim=True)
        return exp_sim


def train_contrastive_epoch(encoder, classifier, train_loader, optim_e, optim_c, scheduler_e, scheduler_c,
                            epoch, contrastive_loss, classifier_loss, args):
    """
    Train each epoch
    """
    total = 0
    correct = 0
    running_loss_e = 0
    running_loss_c = 0
    running_loss = 0

    encoder.to(args.device)
    classifier.to(args.device)

    optim_e.zero_grad()
    optim_c.zero_grad()

    encoder.train()
    classifier.train()

    dataloader = train_loader

    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_p = dataloader.dataset.targets_all['predicted']

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)

    total_updates = int(len(dataloader) * args.batch_factor)
    pbar = tqdm(total=total_updates)
    for batch_idx, batch_data in enumerate(dataloader):
        inputs, labels, data_ix = batch_data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        labels_spurious = [targets_s[ix] for ix in data_ix]

        # Compute contrastive loss first
        outputs_e = encoder(inputs)
        loss_e = contrastive_loss(outputs_e, data_ix, targets_t, targets_p)

        # Then jointly compute classifier loss
        outputs_c = classifier(outputs_e)
        loss_c = classifier_loss(outputs_c, labels)

        loss = (args.contrastive_weight) * loss_e + \
            (1 - args.contrastive_weight) * loss_c

        loss /= args.batch_factor
        loss.backward()

        if ((batch_idx + 1) % args.batch_factor == 0) or (batch_idx + 1 == len(dataloader)):
            # optimizer.step()
            # optimizer.zero_grad()

            optim_e.step()
            if scheduler_e is not None:
                scheduler_e.step()
            optim_c.step()
            if scheduler_c is not None:
                scheduler_c.step()
            optim_e.zero_grad()
            optim_c.zero_grad()
        pbar.update(1)

        data_split = f'Train epoch {epoch}'
        # Save performance
        _, predicted = torch.max(outputs_c.data, 1)
        total += labels.size(0)
        all_correct = (predicted == labels).detach().cpu()
        correct += all_correct.sum().item()
        running_loss_e += loss_e.item()
        running_loss_c += loss_c.item()
        running_loss += loss.item()

        # Save group-wise accuracy
        labels_target = labels.detach().cpu().numpy()
        for ix, s in enumerate(labels_spurious):
            y = labels_target[ix]
            correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
            total_by_groups[int(y)][int(s)] += 1

        pbar.set_description(
            f'{data_split} | Batch Idx: (%d/%d) | Loss: %.3f | Contrastive Loss: %.3f | Classification Loss: %.3f | Acc: %.3f%% (%d/%d) | Contrastive Weight: {args.contrastive_weight}' %
            (batch_idx, len(dataloader), running_loss / (batch_idx + 1), running_loss_e / (batch_idx + 1), running_loss_c / (batch_idx + 1),
                100. * correct / total, correct, total)
        )

        # Clear memory
        inputs = inputs.cpu()
        labels = labels.cpu()
        outputs_e = outputs_e.cpu()
        outputs_c = outputs_c.cpu()
        loss_e = loss_e.cpu()
        loss_c = loss_c.cpu()
        loss = loss.cpu()
        del outputs_e
        del outputs_c
        del inputs
        del labels
        del loss_e
        del loss_c
        del loss
    return running_loss, correct, total, correct_by_groups, total_by_groups


def train_joint_contrastive_epoch(model, train_loader, optim, scheduler,
                                  epoch, contrastive_loss, classifier_loss, args):
    """
    Train each epoch
    """
    total = 0
    correct = 0
    running_loss_e = 0
    running_loss_c = 0
    running_loss = 0

    model.to(args.device)

    optim.zero_grad()

    model.train()

    dataloader = train_loader

    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_p = dataloader.dataset.targets_all['predicted']

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)

    total_updates = int(len(dataloader) * args.batch_factor)
    pbar = tqdm(total=total_updates)
    for batch_idx, batch_data in enumerate(dataloader):
        inputs, labels, data_ix = batch_data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        labels_spurious = [targets_s[ix] for ix in data_ix]

        outputs_e, outputs_c = model.encode_and_classify(inputs)

        # Compute contrastive loss first
        loss_e = contrastive_loss(outputs_e, data_ix, targets_t, targets_p)

        # Then jointly compute classifier loss
        loss_c = classifier_loss(outputs_c, labels)

        loss = (args.contrastive_weight) * loss_e + \
            (1 - args.contrastive_weight) * loss_c

        loss /= args.batch_factor
        loss.backward()

        if ((batch_idx + 1) % args.batch_factor == 0) or (batch_idx + 1 == len(dataloader)):
            # optimizer.step()
            # optimizer.zero_grad()

            optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad()
        pbar.update(1)

        data_split = f'Train epoch {epoch}'
        # Save performance
        _, predicted = torch.max(outputs_c.data, 1)
        total += labels.size(0)
        all_correct = (predicted == labels).detach().cpu()
        correct += all_correct.sum().item()
        running_loss_e += loss_e.item()
        running_loss_c += loss_c.item()
        running_loss += loss.item()

        # Save group-wise accuracy
        labels_target = labels.detach().cpu().numpy()
        for ix, s in enumerate(labels_spurious):
            y = labels_target[ix]
            correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
            total_by_groups[int(y)][int(s)] += 1

        pbar.set_description(
            f'{data_split} | Batch Idx: (%d/%d) | Loss: %.3f | Contrastive Loss: %.3f | Classification Loss: %.3f | Acc: %.3f%% (%d/%d) | Contrastive Weight: {args.contrastive_weight}' %
            (batch_idx, len(dataloader), running_loss / (batch_idx + 1), running_loss_e / (batch_idx + 1), running_loss_c / (batch_idx + 1),
                100. * correct / total, correct, total)
        )

        # Clear memory
        inputs = inputs.cpu()
        labels = labels.cpu()
        outputs_e = outputs_e.cpu()
        outputs_c = outputs_c.cpu()
        loss_e = loss_e.cpu()
        loss_c = loss_c.cpu()
        loss = loss.cpu()
        del outputs_e
        del outputs_c
        del inputs
        del labels
        del loss_e
        del loss_c
        del loss
    return (running_loss, running_loss_e, running_loss_c), correct, total, correct_by_groups, total_by_groups


def initialize_csv_metrics(args):
    train_metrics = {'epoch': [], 'target': [], 'spurious': [],
                     'acc': [], 'avg_acc': [], 'robust_acc': [],
                     'max_avg_acc': [], 'max_robust_acc': []}
    val_metrics = {'epoch': [], 'target': [], 'spurious': [],
                   'acc': [], 'avg_acc': [], 'robust_acc': [],
                   'max_avg_acc': [], 'max_robust_acc': []}
    test_metrics = {'epoch': [], 'target': [], 'spurious': [],
                    'acc': [], 'avg_acc': [], 'robust_acc': [],
                    'max_avg_acc': [], 'max_robust_acc': []}
    args.metrics = {'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics}


def save_metrics(split, epoch, avg_acc, robust_acc, correct_by_groups, total_by_groups, args):
    acc_by_groups = correct_by_groups / total_by_groups
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            args.metrics[split]['epoch'].append(epoch)
            args.metrics[split]['target'].append(yix)
            args.metrics[split]['spurious'].append(aix)
            args.metrics[split]['acc'].append(acc_by_groups[yix][aix])
            args.metrics[split]['avg_acc'].append(avg_acc)
            args.metrics[split]['robust_acc'].append(robust_acc)
            max_avg_acc = np.max(args.metrics[split]['avg_acc'])
            max_robust_acc = np.max(args.metrics[split]['robust_acc'])
            args.metrics[split]['max_avg_acc'].append(max_avg_acc)
            args.metrics[split]['max_robust_acc'].append(max_robust_acc)


def main():
    args = init_args()
    load_dataloaders, visualize_dataset = initialize_data(args)
    init_experiment(args)
    init_logging(args)

    if args.reproduce:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    args.device = (torch.device('cuda:0') if torch.cuda.is_available()
                   and not args.no_cuda else torch.device('cpu'))

    criterion = get_criterion(args, reduction='mean')
    test_criterion = get_criterion(args, reduction='none')

    loaders = load_dataloaders(args, train_shuffle=False)
    train_loader, val_loader, test_loader = loaders
    if args.dataset != 'civilcomments':
        log_data(train_loader.dataset, 'Train dataset:')
        log_data(val_loader.dataset, 'Val dataset:')
        log_data(test_loader.dataset, 'Test dataset:')

    # Stage 1: Train an initial ERM model (or load a pretrained one)
    #          and save its predictions for the training data
    if args.pretrained_spurious_path != '':
        print_header('> Loading stage 1 model')
        stage_1_model = load_pretrained_model(
            args.pretrained_spurious_path, args)
        stage_1_model.eval()
    else:
        print_header('> Training stage 1 model')
        stage_1_model, stage_1_best_model_state_dict = train_stage_1_model(train_loader,
                                                                           args)
        stage_1_model.eval()
    stage_1_predictions = get_stage_1_predictions(
        train_loader, stage_1_model, args)
    train_loader.dataset.targets_all['predicted'] = stage_1_predictions
    train_targets_t = train_loader.dataset.targets_all['target']
    train_targets_s = train_loader.dataset.targets_all['spurious']
    print(
        f'Ground-truth acc: {(stage_1_predictions == train_targets_t).sum() / len(train_targets_t) * 100:<.2f}%')
    print(
        f'Spurious acc:     {(stage_1_predictions == train_targets_s).sum() / len(train_targets_s) * 100:<.2f}%')

    # Stage 2: Train a second contrastive model
    # Resample if specified
    if args.resample_by_group != '':
        train_loader = get_group_resampled_loader(train_loader,
                                                  stage_1_predictions, args)
        if args.dataset != 'civilcomments':
            log_data(train_loader.dataset, 'Train dataset:')

    project = not args.no_projection_head
    # if args.load_encoder != '':
    #     args.checkpoint_name = args.load_encoder
    #     start_epoch = int(args.checkpoint_name.split(
    #         '-cpe=')[-1].split('-')[0])
    #     checkpoint = torch.load(os.path.join(args.model_path,
    #                                          args.checkpoint_name))
    #     print(f'Checkpoint loading from {args.load_encoder}!')
    #     print(f'- Resuming training at epoch {start_epoch}')
    # else:
    # checkpoint = None
    checkpoint = None
    # encoder = ContrastiveNet(args.arch, out_dim=args.projection_dim,
    #                          projection_head=project, task=args.dataset,
    #                          num_classes=args.num_classes,
    #                          checkpoint=checkpoint)
    # classifier = copy.deepcopy(encoder.classifier)
    # for p in encoder.classifier.parameters():
    #     p.requires_grad = False

    # encoder.to(args.device)
    # optimizer_e = get_optim(encoder, args)

    # classifier.to(args.device)
    # optimizer_c = get_optim(classifier, args,
    #                         model_type='classifier')
    model = JointContrastiveNet(args.arch, out_dim=args.projection_dim,
                                projection_head=project, task=args.dataset,
                                num_classes=args.num_classes,
                                checkpoint=checkpoint)
    optimizer = get_optim(model, args)
    classifier_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(args)

    initialize_csv_metrics(args)

    max_robust_val_acc = 0
    max_robust_val_epoch = None

    pbar = tqdm(range(args.max_epoch))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f'Epoch {epoch}')
        else:
            pbar.set_description(
                f'Epoch {epoch} | Train loss: {loss:.3f} | Train contrastive loss: {loss_e:.3f} | Train classification loss: {loss_c:.3f} | Avg val acc: {val_avg_acc:.3f}% | Robust val acc: {val_robust_acc:.3f}%')

        train_outputs = train_joint_contrastive_epoch(model, train_loader, optimizer, scheduler=None,
                                                      epoch=epoch, contrastive_loss=contrastive_loss, classifier_loss=classifier_loss, args=args)
        running_losses, correct, total, correct_by_groups, total_by_groups = train_outputs
        loss, loss_e, loss_c = running_losses
        train_avg_acc = correct / total * 100
        _, train_robust_acc = summarize_acc(correct_by_groups, total_by_groups,
                                            stdout=args.verbose)
        save_metrics('train', epoch, train_avg_acc, train_robust_acc,
                     correct_by_groups, total_by_groups, args)

        val_outputs = evaluate_epoch(model, val_loader, criterion, args)
        val_running_loss, val_correct, val_total, correct_by_groups_v, total_by_groups_v = val_outputs
        val_avg_acc = val_correct / val_total * 100
        _, val_robust_acc = summarize_acc(correct_by_groups_v,
                                          total_by_groups_v,
                                          stdout=args.verbose)
        save_metrics('val', epoch, val_avg_acc, val_robust_acc,
                     correct_by_groups_v, total_by_groups_v, args)

        if val_robust_acc > max_robust_val_acc:
            max_robust_val_acc = val_robust_acc
            max_robust_val_epoch = epoch
            fname = f'cp-{args.experiment_name}-cpe={max_robust_val_epoch}.pt'
            fpath = os.path.join(args.model_path, fname)
            torch.save(model.state_dict(), fpath)
            print(f'Best checkpoint at epoch {epoch}. Saved at {fpath}')

        test_outputs = evaluate_epoch(model, test_loader, criterion, args)
        test_running_loss, test_correct, test_total, correct_by_groups_t, total_by_groups_t = test_outputs
        test_avg_acc = test_correct / test_total * 100
        _, test_robust_acc = summarize_acc(correct_by_groups_t,
                                           total_by_groups_t,
                                           stdout=args.verbose)
        save_metrics('test', epoch, test_avg_acc, test_robust_acc,
                     correct_by_groups_t, total_by_groups_t, args)

        # Save results
        for split in ['train', 'val', 'test']:
            results_path = join(args.results_path,
                                f'{split}-{args.experiment_name}.csv')
            pd.DataFrame(args.metrics[split]).to_csv(results_path)


if __name__ == '__main__':
    main()
