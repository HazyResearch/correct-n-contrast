"""
Should refactor this to actual integrate with the other code

CelebA
```
# Constrastive encoder
python train_supervised_contrast_2.py --arch resnet50_pt --dataset celebA --slice_with rep --pretrained_spurious_path ./model/celebA/celeba_regularized_5.pt --num_anchor 128 --num_positive 128 --num_negative 128 --batch_factor 32 --train_encoder --target_sample_ratio 1 --temperature 0.05 --max_epoch 300 --optim sgd --bs_trn 128 --lr 1e-5 --momentum 0.9 --weight_decay 1e-1 --stopping_window 32 --log_loss_interval 10 --checkpoint_interval 100 --log_visual_interval 200 --log_grad_visual_interval 50 --loss_component nonspurious --lr_outer 1e-4 --n_steps 1 --align_factor 1 --grad_max_epoch 100 --grad_lr 1e-5 --grad_momentum 0.9 --grad_weight_decay 0.1 --grad_bs_trn 128 --grad_slice_with pred_and_rep --verbose --seed 42 --replicate 0 -cs apn --no_projection_head --supervised_linear_scale_up --contrastive_weight 0.75

# No linear scale up
python train_supervised_contrast_2.py --arch resnet50_pt --dataset celebA --slice_with rep --pretrained_spurious_path ./model/celebA/celeba_regularized_5.pt --num_anchor 128 --num_positive 128 --num_negative 128 --batch_factor 32 --train_encoder --target_sample_ratio 1 --temperature 0.05 --max_epoch 300 --optim sgd --bs_trn 128 --lr 1e-5 --momentum 0.9 --weight_decay 1e-1 --stopping_window 32 --log_loss_interval 10 --checkpoint_interval 100 --log_visual_interval 200 --log_grad_visual_interval 50 --loss_component nonspurious --lr_outer 1e-4 --n_steps 1 --align_factor 1 --grad_max_epoch 100 --grad_lr 1e-5 --grad_momentum 0.9 --grad_weight_decay 0.1 --grad_bs_trn 128 --grad_slice_with pred_and_rep --verbose --seed 42 --replicate 0 -cs apn --no_projection_head --contrastive_weight 0.75

python train_supervised_contrastive.py  --arch resnet50_pt --dataset celebA --slice_with rep --rep_cluster_method gmm --pretrained_spurious_path ./model/celebA/celeba_regularized_5.pt  --num_positive 64 --num_negative 64 --batch_factor 32 --contrastive_type contrastive --train_encoder --target_sample_ratio 1.0 --temperature 0.05 --base_temperature 0.05 --max_epoch 1 --lr 1e-5 --momentum 0.9 --weight_decay 1e-1 --stopping_window 32 --log_loss_interval 10 --checkpoint_interval 100 --log_visual_interval 400 --log_grad_visual_interval 50 --loss_component both --lr_outer 1e-5 --n_steps 1 --align_factor 1 --grad_max_epoch 100 --grad_lr 1e-5 --grad_momentum 0.9 --grad_weight_decay 0.1 --grad_bs_trn 128 --grad_slice_with pred_and_rep --grad_rep_cluster_method gmm --verbose --seed 0  --replicate 202 --loss_component nonspurious --contrastive_type contrastive --no_projection_head --retrain_burn_in -1 --replicate 206 --max_epoch 20 --classifier_update_interval 1 --supervised_contrast --replicate 259 --checkpoint_interval 10000 --log_visual_interval 400000 --target_sample_ratio 0.01 --max_epoch 100


python train_supervised_contrast_2.py --arch resnet50_pt --dataset celebA --slice_with rep --rep_cluster_method gmm --pretrained_spurious_path ./model/celebA/celeba_regularized_5.pt --num_positive 64 --num_negative 64 --num_anchor 64 --batch_factor 32 --train_encoder --target_sample_ratio 0.1 --temperature 0.05 --lr 1e-5 --momentum 0.9 --weight_decay 1e-1 --stopping_window 32 --log_loss_interval 10 --checkpoint_interval 100000 --log_visual_interval 400000 --verbose --no_projection_head --contrastive_weight 0.75 -cs apn --seed 0  --replicate 2 --num_negative_easy 64


r-cc-celebA-nph-na=64-np=64-nn=64-nne=64-tsr=0.1-t=0.05-bf=32-cw=0.75-me=300-bst=128-o=sgd-lr=1e-05-mo=0.9-wd=0.001-s=42-r=0.csv

python train_supervised_contrast_2.py --arch resnet50_pt --dataset waterbirds --pretrained_spurious_path "./model/waterbirds/wb_regularized_model.pt" --train_encoder --num_anchor 17 --num_positive 17 --num_negative 17 --batch_factor 32 --optim sgd --lr 1e-3 --momentum 0.9 --weight_decay 1 --grad_lr 1e-4 --grad_momentum 0.9 --grad_weight_decay 1 --grad_bs_trn 128 --grad_max_epoch 300 --loss_component nonspurious --grad_slice_with pred_and_rep --grad_rep_cluster_method gmm --target_sample_ratio 1 --lr 1e-4 --temperature 0.1 --max_epoch 300 --no_projection_head --supervised_linear_scale_up --contrastive_weight 0.75 --log_visual_interval 10000 --checkpoint_interval 10000 --log_loss_interval 10 -cs apn --replicate 0 --seed 0

python train_supervised_contrast_2.py --arch resnet50_pt --dataset waterbirds --pretrained_spurious_path "./model/waterbirds/wb_regularized_model.pt" --train_encoder --num_anchor 1 --num_positive 17 --num_negative 17 --batch_factor 32 --optim sgd --lr 1e-3 --momentum 0.9 --weight_decay 1 --grad_lr 1e-4 --grad_momentum 0.9 --grad_weight_decay 1 --grad_bs_trn 128 --grad_max_epoch 300 --loss_component nonspurious --grad_slice_with pred_and_rep --grad_rep_cluster_method gmm --target_sample_ratio 1 --lr 1e-4 --temperature 0.1 --max_epoch 300 --no_projection_head --contrastive_weight 0.75 --log_visual_interval 10000 --checkpoint_interval 10000 --log_loss_interval 10 --replicate 0 --seed 0

python train_supervised_contrast_2.py --arch cnn --dataset colored_mnist --data_cmap hsv --test_shift random -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9 --p_correlation 0.99 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 --slice_with rep --rep_cluster_method gmm --max_epoch_s 5 --num_anchor 32 --num_positive 32 --num_negative 32 --batch_factor 32 --target_sample_ratio 1 --temperature 0.05 --max_epoch 3 --optim sgd --lr 1e-2 --momentum 0.9 --weight_decay 1e-3 --bs_trn 32 --bs_val 32 --no_cuda --num_workers 0 --no_projection_head --train_encoder --lr_scheduler_classifier linear_decay --lr_scheduler linear_decay --classifier_update_interval 1 --log_loss_interval 10 --checkpoint_interval 10000 --log_visual_interval 40000 --verbose --contrastive_weight 0.5 -cs apn --seed 42 --replicate 0 

# Colored MNIST
python train_supervised_contrast_2.py --no_cuda --arch cnn --dataset colored_mnist --data_cmap hsv --test_shift random -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9 --p_correlation 0.99 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 --slice_with rep --rep_cluster_method gmm --max_epoch_s 5 --num_anchor 32 --num_positive 32 --num_negative 32 --num_negative_easy 32 --batch_factor 32 --target_sample_ratio 1 --temperature 0.05 --max_epoch 3 --optim sgd --lr 1e-2 --momentum 0.9 --weight_decay 1e-3 --bs_trn 32 --bs_val 32 --no_cuda --num_workers 0 --no_projection_head --train_encoder --lr_scheduler_classifier linear_decay --lr_scheduler linear_decay --classifier_update_interval 1 --log_loss_interval 10 --checkpoint_interval 10000 --log_visual_interval 40000 --verbose --contrastive_weight 0.5 -cs apn --seed 42 --replicate 12 


python -W ignore train_supervised_contrast_2.py --arch bert-base-uncased_pt --dataset civilcomments --slice_with rep --rep_cluster_method gmm --pretrained_spurious_path ./model/civilcomments/config/cp-a=bert-base-uncased_pt-d=civilcomments-tm=2s2s_spur-me=2-o=sgd-bs_trn=16-lr=1e-05-mo=0.9-wd=0.01-rc=0-cgn=0-s=42-cpe=2-cpre=0-cpb=-1.pth.tar --num_positive 16 --num_negative 16 --num_anchor 16 --batch_factor 32 --num_negative_easy 16 --train_encoder --target_sample_ratio 0.1 --temperature 0.1 --max_epoch 5 --optim AdamW --lr 1e-5 --log_loss_interval 10 --checkpoint_interval 10000 --log_visual_interval 400000 --verbose --seed 0 --replicate 2 --clip_grad_norm --no_projection_head --contrastive_weight 0.75 -cs apn --seed 0  --replicate 22  --balance_targets --weight_decay 1e-2
```
"""

import os
import sys
import copy
import argparse
import importlib

import torch
import torch.nn.functional as f
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

# Data
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from datasets import get_data_args, train_val_split, get_resampled_indices, get_resampled_set, initialize_data
# Logging and training
from train import train_model, test_model
from utils import print_header, init_experiment, update_contrastive_experiment_name
from utils.logging import Logger, log_args, summarize_acc, initialize_csv_metrics, log_data
from utils.visualize import plot_confusion, plot_data_batch
from utils.metrics import compute_resampled_mutual_info, compute_mutual_info_by_slice
# Model
from network import get_net, get_optim, get_criterion, load_pretrained_model, save_checkpoint
from network import get_output, backprop_, get_bert_scheduler, _get_linear_schedule_with_warmup
# U-MAPS
from activations import visualize_activations
# Contrastive
from contrastive_supervised_loader import prepare_contrastive_points, load_contrastive_data, adjust_num_pos_neg_
# Testing
from contrastive_loader import prepare_contrastive_points as prepare_contrastive_points_old
from contrastive_loader import load_contrastive_data as load_contrastive_data_old

from contrastive_network import ResNetSimCLR, RobustSimCLR
from contrastive_network import ContrastiveLoss, TripletLoss, RobustContrastiveLoss
from slice import compute_pseudolabels, compute_slice_indices
from contrastive_slice import train_spurious_model, get_resampled_sliced_data_indices, visualize_slice_stats
# Alternative -> should debug later
from slice import train_spurious_model
## Alternative slicing by UMAP clustering
from contrastive_slice import compute_slice_indices_by_rep, combine_data_indices
# Grad-aligned Updates
from contrastive_grad_update import train_grad_aligned_model, compute_grad_slices

import transformers
transformers.logging.set_verbosity_error()


def init_args(args):
    args.supervised_contrast = True
    args.prioritize_spurious_pos = False
    args.full_contrastive = False
    args.contrastive_type = 'cc'
    
    # Metrics
    args.compute_auroc = False  # Turn True for certain datasets, e.g. ISIC, CXR8
    if args.dataset in ['isic', 'cxr8']:
        args.compute_auroc = True

    # Model
    args.model_type = f'{args.arch}_2s2s_s1'
    args.criterion = 'cross_entropy'
    args.pretrained = False
    
    ## BERT Defaults
    args.max_grad_norm = 1.0
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    ### Keep these the same for the spurious model
    args.max_grad_norm_s = 1.0
    args.adam_epsilon_s = 1e-8
    args.warmup_steps_s = 0
    ### And the same for grad-aligned finetuning
    args.grad_max_grad_norm = 1.0
    args.grad_adam_epsilon = 1e-8
    args.grad_warmup_steps = 0

    args.device = torch.device('cuda:0') if torch.cuda.is_available() and not args.no_cuda else torch.device('cpu')
    print(args.device)
    
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
    
    # Not actually applied here
    args.val_split = 0.1
    args.spurious_train_split = 0.2
    args.subsample_groups = False
    # args.flipped = False
    # args.test_cmap = ''
    args.train_method = 'sc'  # Because "slicing" by U-MAP, retrain
    
    if args.erm:
        args.train_method += '-erm'
        
    if args.single_pos:
        args.train_method += '-sp'
        
    if args.finetune_epochs > 0:
        args.train_method += '-fce={args.finetune_epochs}'
        
    if args.freeze_encoder:
        args.train_method += '-f'
        
    if args.grad_align:
        args.train_method += '-ga'
    
    # Save accuracies
    args.max_robust_acc = -1
    args.max_robust_epoch = -1
    args.max_robust_group_acc = (None, None)
    
    
def update_args(args):
    args.experiment_name = f'{args.contrastive_type}'
    
    if (args.replicate - 1) % 2 == 0:
        args.num_anchor = 1
        load_contrastive_data = load_contrastive_data_old
        prepare_contrastive_points = prepare_contrastive_points_old
    
    if args.dataset == 'colored_mnist':
        args.experiment_name += f'-cmnist_p{args.p_correlation}-bs_trn_s={args.bs_trn_s}'
    else:
        args.experiment_name += f'-{args.dataset}'

    if args.no_projection_head:
        args.experiment_name += f'-nph'
        
    args.experiment_name += f'-sw={args.slice_with[:2]}'
    args.experiment_name += f'-na={args.num_anchor}-np={args.num_positive}-nn={args.num_negative}-nne={args.num_negative_easy}'
    if args.weight_anc_by_loss:
        args.experiment_name += f'-at={args.anc_loss_temp}'
    if args.weight_pos_by_loss:
        args.experiment_name += f'-pt={args.pos_loss_temp}'
    if args.weight_neg_by_loss:
        args.experiment_name += f'-nt={args.neg_loss_temp}'

    args.experiment_name += f'-tsr={args.target_sample_ratio}-t={args.temperature}'

    if args.hard_negative_factor > 0:
        args.experiment_name += f'-hnf={args.hard_negative_factor}'

    if args.balance_targets:
        args.experiment_name += '-bt'
        
    if args.resample_class != '':
        args.experiment_name += f'-rs={args.resample_class[0]}s'

    args.experiment_name += f'-bf={args.batch_factor}-cw={args.contrastive_weight}'

    if args.supervised_linear_scale_up:
        args.experiment_name += '-slsu'
        
    args.experiment_name += f'-sud={args.supervised_update_delay}'
        
    # Not necessary anymore
#     args.experiment_name += f'-cs='
#     for c in args.contrastive_samples:
#         args.experiment_name += f'{c}'

    if args.single_pos:
        args.experiment_name += '-sp'
        
    if args.finetune_epochs > 0:
        args.experiment_name += f'-fce={args.finetune_epochs}'
        
    if args.freeze_encoder:
        args.experiment_name += '-f'

    model_params = f'-me={args.max_epoch}-bst={args.bs_trn}-o={args.optim}-lr={args.lr}-mo={args.momentum}-wd={args.weight_decay}'
#     if args.weight_decay_c != args.weight_decay:  # May have higher L2 reg for classifier
    model_params += f'-wdc={args.weight_decay_c}'
    if args.lr_scheduler != '':
        model_params += f'-lrs={args.lr_scheduler[:3]}'
    if args.lr_scheduler_classifier != '':
        model_params += f'-clrs={args.lr_scheduler[:3]}'
    
    args.experiment_name += model_params

    args.experiment_name += f'-s={args.seed}-r={args.replicate}'
    print(f'Updated experiment name: {args.experiment_name}')
    
# --------------    
# Training Utils
# --------------
def compute_kl_loss(positive_kl, negative_kl, args):
    kl_loss = (args.kl_pos_factor * positive_kl 
               - args.kl_neg_factor * negative_kl)
    kl_loss = f.relu(kl_loss)
    return kl_loss


def free_gpu(tensors, delete):
    # tensor = tensor.to(torch.device('cpu'))
    for tensor in tensors:
        tensor = tensor.detach().cpu()
        if delete:
            del tensor

            
def compute_outputs(inputs, encoder, classifier, args, 
                    labels=None, compute_loss=False,
                    cross_entropy_loss=None):
    inputs = inputs.to(args.device)
    outputs = encoder.encode(inputs)
    if args.replicate in range(10, 20):
        noise = ((0.01 ** 0.5) * torch.randn(*outputs.shape)).to(args.device)
        outputs = outputs + noise
    
    outputs = classifier(outputs)
    loss = torch.zeros(1)
    
    if compute_loss:
        assert labels is not None; cross_entropy_loss is not None
        labels = labels.to(args.device)
        loss = cross_entropy_loss(outputs, labels)
        if args.arch == 'bert-base-uncased_pt':
            return outputs, loss
        free_gpu([labels], delete=True)
        
    free_gpu([inputs], delete=True)
    return outputs, loss
    
    
def load_encoder_state_dict(model, state_dict, contrastive_train=False):
    # Remove 'backbone' prefix for loading into model
    if contrastive_train:
        log = model.load_state_dict(state_dict, strict=False)
        for k in list(state_dict.keys()):
            print(k)
    else:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):  
                # Corrected for CNN
                if k.startswith('backbone.fc1') or k.startswith('backbone.fc2'):
                    state_dict[k[len("backbone."):]] = state_dict[k]
                # Should also be corrected for BERT models
                elif k.startswith('backbone.fc') or k.startswith('backbone.classifier'):
                    pass
                else:
                    state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
    print(f'log.missing_keys: {log.missing_keys}')
#     if not contrastive_train:
#         assert log.missing_keys == ['fc.weight', 'fc.bias']
    return model


def recompute_slices_with_resampling(dataloader, slice_model, 
                                     sampling, test_criterion,
                                     seed, args, split='Train'):
    """
    Unsure how this differs exactly from balance targets, but it does?
    """
    # Resample indices by class
    resampled_indices = get_resampled_indices(dataloader,
                                              args,
                                              args.resample_class,
                                              seed)
    dataset_resampled = get_resampled_set(dataloader.dataset,
                                          resampled_indices, 
                                          copy_dataset=True)
    dataloader = DataLoader(dataset_resampled,
                            batch_size=args.bs_trn,
                            shuffle=False,
                            num_workers=args.num_workers)
    if args.dataset != 'civilcomments':
        log_data(dataloader.dataset, f'Resampled {split} dataset:')
    
    # Compute slices for contrastive batch sampling
    slice_model.eval()
    slice_model.to(args.device)
    slice_outputs = compute_slice_outputs(slice_model, dataloader,
                                          test_criterion, args)
    # sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
    for _, p in slice_model.named_parameters():
        p = p.to(torch.device('cpu'))
    slice_model.to(torch.device('cpu'))
    
    # sliced_data_indices, sliced_data_correct, sliced_data_losses 
    return slice_outputs


def finetune_model(encoder, criterion, test_criterion, dataloaders, 
                   slice_model, args):
    train_loader, val_loader, test_loader = dataloaders
    model = get_net(args)
    state_dict = encoder.to(torch.device('cpu')).state_dict()
    model = load_encoder_state_dict(model, state_dict)
    args.model_type = 'finetune'
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias', 
                            'backbone.fc.weight', 
                            'backbone.fc.bias']:
                param.requires_grad = False
        # Extra checking
        params = list(filter(lambda p: p.requires_grad, 
                             model.parameters()))
        assert len(params) == 2
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                print(name)
        args.model_type += f'-fe'
        
    optim = get_optim(model, args, model_type='classifier')
    if args.replicate in np.arange(60, 70):
        if args.replicate > 64:
            args.subsample_labels = False
            args.supersample_labels = True
            args.model_type += f'-ss'
        else:
            args.subsample_labels = True
            args.supersample_labels = False
            args.model_type += '-us'
            
        slice_model.to(args.device)
        slice_model.eval()
        slice_outputs = compute_slice_outputs(slice_model,
                                              train_loader,
                                              test_criterion, 
                                              args)
#         slice_outputs = compute_slice_indices(slice_model,
#                                               train_loader,
#                                               test_criterion,
#                                               args.bs_trn, args,
#                                               resample_by='class',
#                                               loss_factor=1)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
        slice_model.to(torch.device('cpu'))
        indices = np.hstack(sliced_data_indices)
        heading = f'Finetuning on aggregated slices'
        print('-' * len(heading))
        print(heading)
        sliced_val_loader = val_loader
        sliced_train_sampler = SubsetRandomSampler(indices)
        sliced_train_loader = DataLoader(train_loader.dataset,
                                         batch_size=args.bs_trn,
                                         sampler=sliced_train_sampler,
                                         num_workers=args.num_workers)
        args.model_type = '2s2s_ss'
        outputs = train_model(model, optim, criterion,
                              sliced_train_loader,
                              sliced_val_loader, args, 0,
                              args.finetune_epochs, True, 
                              test_loader, test_criterion)
    else:
        args.model_type += f'-erm'
        heading = f'Finetuning on original dataset'
        print('-' * len(heading))
        print(heading)
        # Shuffle train_loader
        train_indices = np.arange(len(train_loader.dataset.targets))
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader_ = DataLoader(train_loader.dataset,
                                   batch_size=args.bs_trn,
                                   sampler=train_sampler,
                                   num_workers=args.num_workers)
        
        outputs = train_model(model, optim, criterion,
                              train_loader_, val_loader, args, 0,
                              args.finetune_epochs, True, 
                              test_loader, test_criterion)
    model, max_robust_metrics, all_acc = outputs
    return model


def train_epoch(encoder, classifier, dataloader,
                optim_e, optim_c, scheduler_e, scheduler_c,
                epoch, test_loader, contrastive_loss,
                cross_entropy_loss, args):
    """
    Train contrastive epoch
    """
    encoder.to(args.device)
    classifier.to(args.device)
    
    # Added this 5/18 -
    optim_e.zero_grad()
    optim_c.zero_grad()
    # -----------------
    contrastive_weight = args.contrastive_weight
    loss_compute_size = int(args.num_anchor + args.num_negative 
                            + args.num_positive + args.num_negative_easy)
    epoch_losses = []
    epoch_losses_contrastive = []
    epoch_losses_cross_entropy = []
    epoch_losses_kl = []
    
    # encoder.eval()  # Turn off batchnorm?
    # CHECK THIS
    if args.replicate in [44] or args.seed in [420]:
        encoder.train()
    else:
        encoder.eval()
    classifier.train()
    
    total_updates = int(len(dataloader) * args.batch_factor)
    pbar = tqdm(total=total_updates)
    for batch_ix, batch_data in enumerate(dataloader):
        
        batch_loss = 0
        batch_loss_contrastive = 0
        batch_loss_cross_entropy = 0
        batch_loss_kl = 0
        
        batch_count = 0
        
        # Setup main contrastive batch
        all_batch_inputs, all_batch_labels, all_batch_indices = batch_data
        batch_inputs = torch.split(all_batch_inputs, 
                                   loss_compute_size)
        batch_labels = torch.split(all_batch_labels, 
                                   loss_compute_size)
        batch_indices = np.split(all_batch_indices, len(batch_inputs))
        
        if args.supervised_linear_scale_up:
            supervised_weight = ((1 - args.contrastive_weight) * 
                                 ((epoch * len(dataloader) + batch_ix) * 
                                 args.supervised_step_size))
        elif epoch < args.supervised_update_delay:
            supervised_weight = 0
        else:
            supervised_weight = 1 - args.contrastive_weight

        # print(f'len(batch_inputs): {len(batch_inputs)}')
        for ix, batch_input in enumerate(batch_inputs):
            neg_start_ix = args.num_anchor + args.num_positive
            neg_end_ix = neg_start_ix + args.num_negative
            
            inputs_a  = batch_input[:args.num_anchor]
            inputs_p  = batch_input[args.num_anchor:neg_start_ix]
            inputs_n  = batch_input[neg_start_ix:neg_end_ix]
            inputs_ne = batch_input[-args.num_negative_easy:]

            labels_a  = batch_labels[ix][:args.num_anchor]
            labels_p  = batch_labels[ix][args.num_anchor:neg_start_ix]
            labels_n  = batch_labels[ix][neg_start_ix:neg_end_ix]
            labels_ne = batch_labels[ix][-args.num_negative_easy:]
            
            # Just do contrastive loss against first anchor for now
            inputs_a_ = [inputs_a[0]]
            for anchor_ix, input_a in enumerate(inputs_a_):
                contrastive_batch = torch.vstack((input_a.unsqueeze(0),
                                                  inputs_p, inputs_n))
                # Compute contrastive loss
                loss = contrastive_loss(encoder, contrastive_batch)
                loss *= ((1 - supervised_weight) / 
                         (len(inputs_a_) * len(batch_inputs)))
                if args.replicate in [2, 8] and args.num_negative_easy > 0:
                    loss *= 0.5
                # smh - so annoying... Ok if
                elif args.replicate in np.arange(50, 100) and (args.replicate % 2) == 0:
                    loss *= 0.5
                
                loss.backward()
                contrastive_batch = contrastive_batch.detach().cpu()
                
                batch_loss += loss.item()
                batch_loss_contrastive += loss.item()
                free_gpu([loss], delete=True)
                
                # Compute the provable way
                if args.num_negative_easy > 0:
                    contrastive_batch = torch.vstack(
                        (inputs_p[0].unsqueeze(0), inputs_a, inputs_ne)
                    )
                    # Compute contrastive loss
                    loss = contrastive_loss(encoder, contrastive_batch)
                    loss *= ((1 - supervised_weight) / 
                             (len(inputs_a_) * len(batch_inputs)))
                    
                    if args.replicate in [2, 8]:
                        loss *= 0.5   
                    # smh - so annoying... Ok anything other than those is not loss *= 0.5
                    elif args.replicate in np.arange(50, 100) and (args.replicate % 2) == 0:
                        loss *= 0.5

                    loss.backward()
                    contrastive_batch = contrastive_batch.detach().cpu()

                    batch_loss += loss.item()
                    batch_loss_contrastive += loss.item()
                    free_gpu([loss], delete=True)
                    
                if args.finetune_epochs > 0:
                    continue
                
                # A bit gross?
                if anchor_ix + 1 == len(inputs_a_):
                    # Added 5/13 - set replicate > 10 for these
                    input_list = [inputs_a, inputs_p, inputs_n, inputs_ne]
                    label_list = [labels_a, labels_p, labels_n, labels_ne]
                    
                    if args.replicate > 30 and args.replicate < 40:
                        input_list = [inputs_a, inputs_p, inputs_n]
                        label_list = [labels_a, labels_p, labels_n]
                    min_input_size = np.min([len(x) for x in input_list])
                    contrast_inputs = torch.cat([x[:min_input_size] for x in input_list])
                    contrast_labels = torch.cat([l[:min_input_size] for l in label_list])
                    if loss_compute_size <= args.bs_trn:
                        # Can play around here with different inputs, e.g. some portion of the positives and negatives too?
                        output, loss = compute_outputs(contrast_inputs, 
                                                       encoder, classifier,
                                                       args, 
                                                       contrast_labels, 
                                                       True,
                                                       cross_entropy_loss)
                        loss *= (supervised_weight / len(batch_inputs))
                        loss.backward()
                        batch_loss += loss.item()
                        batch_loss_cross_entropy += loss.item()
                        free_gpu([loss], delete=True)
                    else:
                        # contrast_inputs = torch.stack((inputs_a, inputs_p, inputs_n), dim=1).reshape(-1, *list(inputs_a.shape)[1:])
                        # contrast_inputs = torch.split(contrast_inputs, args.bs_trn)

                        # contrast_labels = torch.stack((labels_a, labels_p, labels_n), dim=1).reshape(-1)
                        # contrast_labels = torch.split(contrast_labels, args.bs_trn)
                        
                        # Shuffle together and split
                        # contrast_inputs = torch.cat((inputs_a, inputs_p, inputs_n))
                        # contrast_labels = torch.cat((labels_a, labels_p, labels_n))
                        
                        shuffle_ix = np.arange(contrast_inputs.shape[0])
                        np.random.shuffle(shuffle_ix)
                        contrast_inputs = contrast_inputs[shuffle_ix]
                        contrast_labels = contrast_labels[shuffle_ix]
                        
                        contrast_inputs = torch.split(contrast_inputs, args.bs_trn)
                        contrast_labels = torch.split(contrast_labels, args.bs_trn)

                        for cix, contrast_input in enumerate(contrast_inputs):
                            
                            weight = contrast_input.shape[0] / len(shuffle_ix)
                            output, loss = compute_outputs(contrast_input, 
                                                           encoder,
                                                           classifier,
                                                           args,
                                                           contrast_labels[cix], 
                                                           True, cross_entropy_loss)
                            loss *= (supervised_weight * weight / len(batch_inputs))
                            loss.backward()
                        
                            batch_loss += loss.item()
                            batch_loss_cross_entropy += loss.item()

                            free_gpu([loss, output], delete=True)
                    
                batch_count += 1
                
            pbar.update(1)

        if args.arch == 'bert-base-uncased_pt':
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                               args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),
                                               args.max_grad_norm)
        if args.finetune_epochs > 0:
            optim_e.step()
            if scheduler_e is not None:
                scheduler_e.step()
            optim_e.zero_grad()
        else:
            optim_e.step()
            if scheduler_e is not None:
                scheduler_e.step()
            optim_c.step()
            if scheduler_c is not None:
                scheduler_c.step()
            optim_e.zero_grad()
            if args.replicate > 50 or args.replicate in [8, 4, 36, 44]:
                optim_c.zero_grad()  # For replicate < 50, this was also optim_c.zero_grad before, i.e. not zeroing grads?
            else:
                optim_c.zero_grad
        
        epoch_losses.append(batch_loss)
        epoch_losses_contrastive.append(batch_loss_contrastive)
        epoch_losses_cross_entropy.append(batch_loss_cross_entropy)
        # epoch_losses_kl.append(batch_loss_kl)
        
        if (batch_ix + 1) % args.log_loss_interval == 0:
            print_output  = f'Epoch {epoch:>3d} | Batch {batch_ix:>4d} | '
            print_output += f'Loss: {batch_loss:<.4f} (Epoch Avg: {np.mean(epoch_losses):<.4f}) | '
            print_output += f'CL: {batch_loss_contrastive:<.4f} (Epoch Avg: {np.mean(epoch_losses_contrastive):<.4f}) | '
            print_output += f'CE: {batch_loss_cross_entropy:<.4f}, (Epoch Avg: {np.mean(epoch_losses_cross_entropy):<.4f}) | '
            # print_output += f'KL: {batch_loss_kl:<.4f}, Epoch Avg: {np.mean(epoch_losses_kl):<.4f}'
            print_output += f'SW: {supervised_weight:<.4f}'
            print(print_output)
            
        if (batch_ix + 1) % args.checkpoint_interval == 0 or (batch_ix + 1) == len(dataloader):
#             checkpoint_name = save_checkpoint(encoder, optim_e,
#                                               np.mean(epoch_losses), 
#                                               epoch, batch_ix, args, 
#                                               replace=True,
#                                               retrain_epoch=-1,
#                                               identifier='enc')
            
#             args.checkpoint_name = checkpoint_name
#             checkpoint_name = save_checkpoint(classifier, optim_c,
#                                               np.mean(epoch_losses),
#                                               epoch, batch_ix, args,
#                                               replace=True,
#                                               retrain_epoch=-1,
#                                               identifier='cls')
#             args.checkpoint_name = checkpoint_name
            model = get_net(args)
            state_dict = encoder.to(torch.device('cpu')).state_dict()
            model = load_encoder_state_dict(model, state_dict)
            if 'bert' in args.arch:
                model.classifier = classifier
            else:
                model.fc = classifier
                # model.classifier = classifier
            checkpoint_name = save_checkpoint(model, None,
                                              np.mean(epoch_losses),
                                              epoch, batch_ix, args,
                                              replace=True,
                                              retrain_epoch=-1,
                                              identifier='fm')
            args.checkpoint_name = checkpoint_name
            
    epoch_losses = (epoch_losses,
                    epoch_losses_contrastive,
                    epoch_losses_cross_entropy,
                    epoch_losses_kl)
    return encoder, classifier, epoch_losses


def evaluate_model(model, dataloaders, modes, test_criterion, args, epoch):
    """
    Args:
        - modes (str[]): ['Training', 'Testing']
    """
    # Assume test dataloader is last
    for dix, dataloader in enumerate(dataloaders):
        test_outputs = test_model(model, dataloader, test_criterion, 
                                  args, epoch, modes[dix])
        test_running_loss, test_correct, test_total, correct_by_groups, total_by_groups, correct_indices, all_losses, loss_by_groups = test_outputs
    
    robust_acc = summarize_acc(correct_by_groups, total_by_groups,
                               stdout=False)
    print(f'Robust acc: {robust_acc}')
    print(f'Max robust acc: {args.max_robust_acc}')
    
    if robust_acc > args.max_robust_acc:
        print(f'New max robust acc: {robust_acc}')
        args.max_robust_acc = robust_acc
        args.max_robust_epoch = epoch
        args.max_robust_group_acc = (correct_by_groups, total_by_groups)
        
        print(f'- Saving best checkpoint at epoch {epoch}')
        checkpoint_name = save_checkpoint(model, None,
                                          robust_acc,  # override loss
                                          epoch, -1, args,
                                          replace=True,
                                          retrain_epoch=-1,
                                          identifier='fm_b')
        args.checkpoint_name = checkpoint_name
        
        if 'bert' not in args.arch:
            # Visualize highest confidence and random incorrect test samples
            max_loss_indices = np.argsort(all_losses)[-64:]
            plot_data_batch([dataloader.dataset.__getitem__(i)[0] for i in max_loss_indices],
                            mean=args.image_mean, std=args.image_std, nrow=8,
                            title='Highest Confidence Incorrect Test Samples',
                            args=args, save=True,
                            save_id=f'ic_hc-e{epoch}', ftype=args.img_file_type)
            false_indices = np.where(
                np.concatenate(correct_indices, axis=0) == False)[0]
            plot_data_batch([dataloader.dataset.__getitem__(i)[0] for i in false_indices[:64]],
                            mean=args.image_mean, std=args.image_std, nrow=8,
                            title='Random Incorrect Test Samples',
                            args=args, save=True,
                            save_id=f'ic_rd-e{epoch}', ftype=args.img_file_type)
    
    save_path = os.path.join(args.results_path,
                             f'r-{args.experiment_name}.csv')
    pd.DataFrame(args.test_metrics).to_csv(save_path, index=False)
    print(f'Test metrics saved to {save_path}!')
    
    plt.plot(args.test_metrics['robust_acc'], label='robust acc.')
    plt.plot(args.test_metrics['max_robust_acc'], label='max robust acc.')
    plt.title(f'Worst-group test accuracy')
    plt.legend()
    figpath = os.path.join(args.image_path, f'ta-{args.experiment_name}.png')
    plt.savefig(figpath)
    plt.close()


def run_final_evaluation(model, test_loader, test_criterion, args, epoch,
                         visualize_representation=True):
    test_outputs = test_model(model, test_loader, test_criterion, 
                              args, epoch, 'Testing')
    test_running_loss, test_correct, test_total, correct_by_groups, total_by_groups, correct_indices, all_losses, loss_by_groups = test_outputs
    # Summarize accuracies by group and plot confusion matrix
    if epoch + 1 == args.max_epoch:
        print('Final:')
        robust_acc = summarize_acc(correct_by_groups, total_by_groups,
                                   stdout=False)
        print(f'Robust acc: {robust_acc}')
    
        if robust_acc > args.max_robust_acc:
            print(f'New max robust acc: {robust_acc}')
            args.max_robust_acc = robust_acc
            args.max_robust_epoch = epoch
            args.max_robust_group_acc = (correct_by_groups, total_by_groups)
            
            checkpoint_name = save_checkpoint(model, None,
                                              robust_acc,  # override loss
                                              epoch, -1, args,
                                              replace=True,
                                              retrain_epoch=-1,
                                              identifier='fm_lb')
        
    
        save_id = f'{args.train_method}-epoch'
        plot_confusion(correct_by_groups, total_by_groups, save_id=save_id,
                       save=True, ftype=args.img_file_type, args=args)
    # Save results
    try:
        save_path = os.path.join(args.results_path,
                                 f'r-{args.experiment_name}.csv')
        pd.DataFrame(args.test_metrics).to_csv(save_path, index=False)
    except Exception as e:
        print(e)
        save_path = f'r-{args.experiment_name}.csv'
        pd.DataFrame(args.test_metrics).to_csv(save_path, index=False)
        
    if 'bert' not in args.arch and visualize_representation:
        # Visualize highest confidence and random incorrect test samples
        max_loss_indices = np.argsort(all_losses)[-64:]
        plot_data_batch([test_loader.dataset.__getitem__(i)[0] for i in max_loss_indices],
                        mean=args.image_mean, std=args.image_std, nrow=8,
                        title='Highest Confidence Incorrect Test Samples',
                        args=args, save=True,
                        save_id='ic_hc', ftype=args.img_file_type)
        false_indices = np.where(
            np.concatenate(correct_indices, axis=0) == False)[0]
        plot_data_batch([test_loader.dataset.__getitem__(i)[0] for i in false_indices[:64]],
                        mean=args.image_mean, std=args.image_std, nrow=8,
                        title='Random Incorrect Test Samples',
                        args=args, save=True,
                        save_id='ic_rd', ftype=args.img_file_type)
        # Visualize U-MAPs of activations
    if visualize_representation and 'bert' not in args.arch:
        suffix = f'(robust acc: {robust_acc:<.3f})'
        save_id = f'{args.contrastive_type[0]}g{args.max_epoch}'
        visualize_activations(model, dataloader=test_loader,
                              label_types=['target', 'spurious', 'group_idx'],
                              num_data=1000, figsize=(8, 6), save=True,
                              ftype=args.img_file_type, title_suffix=suffix,
                              save_id_suffix=save_id, args=args)
        
    
        
        
def compute_slice_outputs(slice_model, train_loader, test_criterion, args):
    if 'rep' in args.slice_with:
        slice_outputs = compute_slice_indices_by_rep(slice_model,
                                                     train_loader,
                                                     cluster_umap=True, 
                                                     umap_components=2,
                                                     cluster_method=args.rep_cluster_method,
                                                     args=args,
                                                     visualize=True)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs

    if 'pred' in args.slice_with:
        slice_outputs_ = compute_slice_indices(slice_model, train_loader, 
                                               test_criterion, 1, 
                                               args, 
                                               resample_by='class',
                                               loss_factor=args.loss_factor,
                                               use_dataloader=True)
        sliced_data_indices_, sliced_data_losses_, sliced_data_correct_, sliced_data_probs_ = slice_outputs_

    if args.slice_with == 'pred_and_rep':
        # Combine the indices
        sliced_data_indices, sliced_data_correct = combine_data_indices(
            [sliced_data_indices, sliced_data_indices_], 
            [sliced_data_correct, sliced_data_correct_])
#         sliced_data_indices_t = sliced_data_indices_t_  # Or could be the other one
    elif args.slice_with == 'pred':
        sliced_data_indices = sliced_data_indices_
        sliced_data_correct = sliced_data_correct_
        sliced_data_losses = sliced_data_losses_
        
    return sliced_data_indices, sliced_data_correct, sliced_data_losses
        

def main():
    parser = argparse.ArgumentParser(description='Compare & Contrast')
    # Model
    parser.add_argument('--arch', choices=['base', 'mlp', 'cnn', 
                                           'resnet50', 'resnet50_pt', 
                                           'resnet34', 'resnet34_pt',
                                           'bert-base-uncased_pt'], required=True)

    parser.add_argument('--bs_trn', type=int, default=128)
    parser.add_argument('--bs_val', type=int, default=128)
    ## Only for MLP
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    # Data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--resample_class', type=str, default='',
                        choices=['upsample', 'subsample', ''],
                        help="Resample datapoints to balance classes")
    
    ## Initial slicing for anchor-positive-negative generation
    parser.add_argument('--slice_with', type=str, default='rep',
                        choices=['rep', 'pred', 'pred_and_rep'])
    parser.add_argument('--rep_cluster_method', type=str, 
                        default='gmm', choices=['kmeans', 'gmm'])
    # parser.add_argument('--retrain_burn_in', type=int, default=300)
    
    ## Set up contrastive batch datapoints
    parser.add_argument('--num_anchor', type=int, default=32)
    parser.add_argument('--num_positive', type=int, default=32)
    parser.add_argument('--num_negative', type=int, default=32)
    parser.add_argument('--num_negative_easy', type=int, default=0)
    ### Sample harder datapoints
    parser.add_argument('--weight_anc_by_loss', default=False, action='store_true')
    parser.add_argument('--weight_pos_by_loss', default=False, action='store_true')
    parser.add_argument('--weight_neg_by_loss', default=False, action='store_true')
    parser.add_argument('--anc_loss_temp', type=float, default=0.5)
    parser.add_argument('--pos_loss_temp', type=float, default=0.5)
    parser.add_argument('--neg_loss_temp', type=float, default=0.5)
    
    parser.add_argument('--data_wide_pos', default=False, action='store_true')
    parser.add_argument('--target_sample_ratio', type=float, default=1)
    parser.add_argument('--balance_targets', default=False, action='store_true')
    parser.add_argument('--additional_negatives', default=False,
                        action='store_true')
    parser.add_argument('--hard_negative_factor', type=float, default=0)
    parser.add_argument('--full_contrastive', default=False,
                        action='store_true')
    
    # Training
    ## Contrastive model
    parser.add_argument('--train_encoder', default=False, action='store_true')
    parser.add_argument('--no_projection_head', default=False, action='store_true')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--batch_factor', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--single_pos', default=False, action='store_true')
    ### Scale up the supervised weight factor
    parser.add_argument('--supervised_linear_scale_up', default=False,
                        action='store_true')
    parser.add_argument('--supervised_update_delay', type=int, default=0)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    ## Classifier
    parser.add_argument('--classifier_update_interval', type=int, default=8)
    ### Supervised
    parser.add_argument('-cs', '--contrastive_samples', type=str, default='apn')
    # Note - this argument isn't actually used now w/ shuffling batches
    ## General training hyperparameters
    parser.add_argument('--optim', type=str, default='sgd', 
                        choices=['AdamW', 'adam', 'sgd'])  # Keep the same for all stages
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_c', type=float, default=-1)
    
    parser.add_argument('--stopping_window', type=int, default=30)
    ### Load pre-trained contrastive model
    parser.add_argument('--load_encoder', type=str, default='')
    ### Freeze encoder layers during second stage training
    parser.add_argument('--freeze_encoder', default=False, 
                        action='store_true')
    parser.add_argument('--finetune_epochs', type=int, default=0)
    ### For BERT, whether to clip grad norm
    parser.add_argument('--clip_grad_norm', default=False, 
                        action='store_true')
    # LR Scheduler -> Only linear decay supported now
    parser.add_argument('--lr_scheduler_classifier', type=str, default='')
    parser.add_argument('--lr_scheduler', type=str, default='')
    
#     # Cross-entropy + KL prediction regularization
#     parser.add_argument('--a_cross_entropy_weight', type=float, default=1)
#     parser.add_argument('--p_cross_entropy_weight', type=float, default=0)
#     parser.add_argument('--n_cross_entropy_weight', type=float, default=0)
#     ## KL factors
#     parser.add_argument('--kl_pos_factor', type=float, default=1)
#     parser.add_argument('--kl_neg_factor', type=float, default=0)
#     ## Total weighting: a * CE + b * KL
#     parser.add_argument('--cross_entropy_weight', type=float, default=0.5)
#     parser.add_argument('--kl_weight', type=float, default=0.5)
    
    
    # Training gradient-aligned model
    parser.add_argument('--grad_align', default=False, action='store_true')
    parser.add_argument('--loss_component', type=str, default='none',
                        choices=['spurious', 'nonspurious', 'both', 'none'])
    parser.add_argument('--grad_slice_with', type=str, default='rep',
                        choices=['rep', 'pred', 'pred_and_rep'])
    parser.add_argument('--grad_rep_cluster_method', type=str, 
                        default='gmm', choices=['kmeans', 'gmm'])
    parser.add_argument('--lr_outer', type=float, default=1e-4)
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--align_factor', type=float, default=1)
    parser.add_argument('--grad_max_epoch', type=int, default=100)
    parser.add_argument('--grad_lr', type=float, default=1e-4)
    parser.add_argument('--grad_momentum', type=float, default=0.9)
    parser.add_argument('--grad_weight_decay', type=float, default=1)
    parser.add_argument('--grad_bs_trn', type=int, default=128)
    ## For BERT, whether to clip grad norm
    parser.add_argument('--grad_clip_grad_norm', default=False, action='store_true')
    ## Actually train with balanced ERM
    parser.add_argument('--erm', default=False, action='store_true')
    
    ## Just train with ERM / load pretrained ERM model
    parser.add_argument('--erm_only', default=False, action='store_true')
    
    ## Training spurious features model
    parser.add_argument('--pretrained_spurious_path', default='', type=str)
    parser.add_argument('--max_epoch_s', type=int, default=1,
                        help="Number of epochs to train initial spurious model")
    parser.add_argument('--bs_trn_s', type=int, default=32,
                        help="Training batch size for core feature model")
    parser.add_argument('--lr_s', type=float, default=1e-3,
                        help="Learning rate for spurious feature model")
    parser.add_argument('--momentum_s', type=float, default=0.9,
                        help="Momentum for spurious feature model")
    parser.add_argument('--weight_decay_s', type=float, default=5e-4,
                        help="Weight decay for spurious feature model")
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
    
    # Set actual default weight_decay for classifier
    if args.weight_decay_c < 0:
        args.weight_decay_c = args.weight_decay
    
    
    init_args(args)
    load_dataloaders, visualize_dataset = initialize_data(args)
    init_experiment(args)
    update_contrastive_experiment_name(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args.mi_resampled = None
    args.image_path = os.path.join(args.image_path, 'contrastive_umaps')
    if not os.path.exists(args.image_path):
        os.makedirs(args.image_path)
    args.device = (torch.device('cuda:0') if torch.cuda.is_available()
               and not args.no_cuda else torch.device('cpu'))
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
    
    criterion = get_criterion(args, reduction='mean')
    criterion_no_reduction = get_criterion(args, reduction='none')
    test_criterion = get_criterion(args, reduction='none')
    
    loaders = load_dataloaders(args, train_shuffle=False)
    train_loader, val_loader, test_loader = loaders
    
    if args.resample_class != '':
        resampled_indices = get_resampled_indices(dataloader=train_loader,
                                                  args=args,
                                                  sampling=args.resample_class,
                                                  seed=args.seed)
        train_set_resampled = get_resampled_set(dataset=train_loader.dataset,
                                                resampled_set_indices=resampled_indices, 
                                                copy_dataset=True)
        train_loader = DataLoader(train_set_resampled,
                                  batch_size=args.bs_trn,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    if args.dataset != 'civilcomments':
        log_data(train_loader.dataset, 'Train dataset:')
        log_data(val_loader.dataset, 'Val dataset:')
        log_data(test_loader.dataset, 'Test dataset:')
        
    # Test
#     encoder = RobustSimCLR(args.arch, out_dim=args.projection_dim, 
#                            projection_head=False, task=args.dataset, 
#                            num_classes=args.num_classes,
#                            checkpoint=None)

    if args.evaluate:
        project = not args.no_projection_head
        assert args.load_encoder != ''
        args.checkpoint_name = args.load_encoder
        start_epoch = int(args.checkpoint_name.split('-cpe=')[-1].split('-')[0])
        checkpoint = torch.load(os.path.join(args.model_path,
                                             args.checkpoint_name))
        print(f'Checkpoint loading from {args.load_encoder}!')
        print(f'- Resuming training at epoch {start_epoch}')
        
        encoder = RobustSimCLR(args.arch, out_dim=args.projection_dim, 
                               projection_head=project,
                               task=args.dataset, 
                               num_classes=args.num_classes,
                               checkpoint=checkpoint)
        classifier = copy.deepcopy(encoder.classifier)
        encoder.to(torch.device('cpu'))
        classifier.to(torch.device('cpu'))
        model = get_net(args)
        state_dict = encoder.to(torch.device('cpu')).state_dict()
        for k in list(state_dict.keys()):
            if k.startswith('fc.') and 'bert' in args.arch:
                state_dict[f'classifier.{k[3:]}'] = state_dict[k]
                # state_dict[k[f'classifier.{k[3:]}']] = state_dict[k]
                del state_dict[k]
        
        model = load_encoder_state_dict(model, state_dict)
        try:
            model.fc = classifier
        except:
            model.classifier = classifier
        run_final_evaluation(model, test_loader, test_criterion,
                             args, epoch=start_epoch, 
                             visualize_representation=True)

        print('Done training')
        print(f'- Experiment name: {args.experiment_name}')
        print_header(f'Max Robust Acc:')
        print(f'Acc: {args.max_robust_acc}')
        print(f'Epoch: {args.max_robust_epoch}')
        summarize_acc(args.max_robust_group_acc[0],
                      args.max_robust_group_acc[1])
        return

        
    # -------------------
    # Slice training data
    # -------------------
    if args.pretrained_spurious_path != '':
        print_header('> Loading spurious model')
        slice_model = load_pretrained_model(args.pretrained_spurious_path, args)
        slice_model.eval()
        args.mode = 'train_spurious'
    else:
        args.mode = 'train_spurious'
        print_header('> Training spurious model')
#         spurious_outputs = train_spurious_model(train_loader, val_loader,  args,
#                                                 test_loader, test_criterion,
#                                                 resample=args.resample_class)
#         slice_model, outputs, _ = spurious_outputs
        args.spurious_train_split = 0.99
        slice_model, outputs, _ = train_spurious_model(train_loader, args)

    slice_model.eval()
    # slice_model.activation_layer = 'avgpool'
    print(f'Pretrained model loaded from {args.pretrained_spurious_path}')
    
    if args.train_encoder is True:  # and not args.load_encoder:
        slice_outputs = compute_slice_outputs(slice_model, train_loader,
                                              test_criterion, args)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
   
        print_header(f'DEBUG', style='top')
        for ix, indices in enumerate(sliced_data_indices):
            train_targets_all = train_loader.dataset.targets_all
            print(f'Slice {ix} ({len(indices)}):')
            for tix, target_value in enumerate(np.unique(train_targets_all['target'])):
                for six, spurious_value in enumerate(np.unique(train_targets_all['spurious'])):
                    group_ix = np.where(np.logical_and(
                        train_targets_all['target'][indices] == target_value,
                        train_targets_all['spurious'][indices] == spurious_value))[0]
                    print(f'- (Target = {tix}, Spurious = {six}): {len(indices[group_ix])}')
        print_header(f'END DEBUG', style='bottom')   
        # End method debugging
        
        print(f'len(sliced_data_indices): {len(sliced_data_indices)}')
        # Report empirical MI(Y | Z_s) = \sum_{z_s} (H(Y) - H(Y | Z_s = z_s))
        print_header('Resampled MI', style='top')
        mi_by_slice = compute_mutual_info_by_slice(train_loader, sliced_data_indices)
        for ix, mi in enumerate(mi_by_slice):
            print(f'H(Y) - H(Y | Z = z_{ix}) = {mi:<.3f} (by slice)')
        mi_resampled = compute_resampled_mutual_info(train_loader, sliced_data_indices)
        print_header(f'H(Y) - H(Y | Z) = {mi_resampled:<.3f}')
        args.mi_resampled = mi_resampled
        
        for _, p in slice_model.named_parameters():
            p = p.to(torch.device('cpu'))
        slice_model.to(torch.device('cpu'))
        
        
        # -------------
        # Train encoder
        # -------------
        args.checkpoint_name = ''
        args.mode = 'contrastive_train'
        start_epoch = 0
        max_epoch = args.max_epoch
        
        contrastive_points = prepare_contrastive_points(sliced_data_indices,
                                                        sliced_data_losses,
                                                        sliced_data_correct,
                                                        train_loader, args)
        slice_anchors, slice_negatives, positives_by_class, all_targets = contrastive_points
        
        adjust_num_pos_neg_(positives_by_class, slice_negatives, args)
        if args.num_negative_easy == 32:  # For now
            args.num_negative_easy = args.num_negative  # Adjust?
        update_args(args)
        
        project = not args.no_projection_head
        if args.load_encoder != '':
            args.checkpoint_name = args.load_encoder
            start_epoch = int(args.checkpoint_name.split('-cpe=')[-1].split('-')[0])
            checkpoint = torch.load(os.path.join(args.model_path,
                                                 args.checkpoint_name))
            print(f'Checkpoint loading from {args.load_encoder}!')
            print(f'- Resuming training at epoch {start_epoch}')
        else:
            checkpoint = None
        
        encoder = RobustSimCLR(args.arch, out_dim=args.projection_dim, 
                               projection_head=project, task=args.dataset, 
                               num_classes=args.num_classes,
                               checkpoint=checkpoint)

        classifier = copy.deepcopy(encoder.classifier)
        for p in encoder.classifier.parameters():
            p.requires_grad = False
            
        print_header(f'Classifier initialized')
        print(f'Testing grad dependence')
        for n, p in classifier.named_parameters():
            print(f'- {n}: {p.requires_grad}')
        print(f'Classifier outputs: {encoder.num_classes}')

        encoder.to(args.device)
        optimizer = get_optim(encoder, args)

        classifier.to(args.device)
        classifier_optimizer = get_optim(classifier, args,
                                         model_type='classifier')
        
        # Dummy scheduler initialization
        if 'bert' in args.arch:
            # num_training_steps = len(dataloader) * n_epochs
            scheduler = get_bert_scheduler(optimizer, n_epochs=1,
                                           warmup_steps=args.warmup_steps, 
                                           dataloader=np.arange(10))
        else:
            if args.lr_scheduler == 'linear_decay':
                scheduler = _get_linear_schedule_with_warmup(optimizer,
                                                             args.warmup_steps,
                                                             num_training_steps=10)
            if args.lr_scheduler_classifier == 'linear_decay':
                classifier_scheduler = _get_linear_schedule_with_warmup(
                    classifier_optimizer, args.warmup_steps, 10)

        cross_entropy_loss = get_criterion(args, reduction='mean')
        contrastive_loss = RobustContrastiveLoss(args)
        
        args.epoch_mean_loss = 1e5
        all_losses = []
        all_losses_cl = []
        all_losses_ce = []
        all_losses_kl = []

        # Get contrastive batches for first epoch
        epoch = 0
        contrastive_dataloader = load_contrastive_data(train_loader, 
                                                       slice_anchors, 
                                                       slice_negatives, 
                                                       positives_by_class,
                                                       epoch+args.seed, 
                                                       args, True)

        if args.supervised_linear_scale_up:
            args.supervised_step_size = (1 / (len(contrastive_dataloader) * 
                                   args.max_epoch))
        else:
            args.supervised_step_size = 0

        initialize_csv_metrics(args)
        for epoch in range(start_epoch, max_epoch):
            encoder.to(args.device)
            classifier.to(args.device)
            
            # Schedulers
            scheduler = None
            classifier_scheduler = None
            total_updates = int(np.round(
                len(contrastive_dataloader) * (max_epoch - start_epoch)))
            last_epoch = int(np.round(epoch * len(contrastive_dataloader)))
            if 'bert' in args.arch:
                # num_training_steps = len(dataloader) * n_epochs
                scheduler = get_bert_scheduler(optimizer, n_epochs=total_updates,
                                               warmup_steps=args.warmup_steps, 
                                               dataloader=contrastive_dataloader,
                                               last_epoch=last_epoch)
            else:
                if args.lr_scheduler == 'linear_decay':
                    scheduler = _get_linear_schedule_with_warmup(optimizer,
                                                                 args.warmup_steps,
                                                                 total_updates,
                                                                 last_epoch)
            if args.lr_scheduler_classifier == 'linear_decay':
                classifier_scheduler = _get_linear_schedule_with_warmup(
                    classifier_optimizer, args.warmup_steps, total_updates, last_epoch)

            train_outputs = train_epoch(encoder, classifier, 
                                        contrastive_dataloader,
                                        optimizer, classifier_optimizer,
                                        scheduler, classifier_scheduler,
                                        epoch, test_loader, 
                                        contrastive_loss, cross_entropy_loss,
                                        args)

            encoder, classifier, epoch_losses = train_outputs
            epoch_loss, epoch_loss_cl, epoch_loss_ce, epoch_loss_kl = epoch_losses
            all_losses.extend(epoch_loss)
            all_losses_cl.extend(epoch_loss_cl)
            all_losses_ce.extend(epoch_loss_ce)
            all_losses_kl.extend(epoch_loss_kl)

            if 'bert' not in args.arch:  # Bug for now
                # Visualize
                suffix = f'(epoch {epoch}, epoch loss: {np.mean(epoch_loss):<.3f}, train)'
                save_id = f'{args.contrastive_type[0]}-tr-e{epoch}-final'
                visualize_activations(encoder, dataloader=train_loader,
                                      label_types=['target', 'spurious', 'group_idx'],
                                      num_data=1000, figsize=(8, 6), save=True,
                                      ftype=args.img_file_type, title_suffix=suffix,
                                      save_id_suffix=save_id, args=args,
                                      annotate_points=None)
                suffix = f'(epoch {epoch}, epoch loss: {np.mean(epoch_loss):<.3f}, test)'
                save_id = f'{args.contrastive_type[0]}-e{epoch}-final'
                visualize_activations(encoder, dataloader=test_loader,
                                      label_types=['target', 'spurious', 'group_idx'],
                                      num_data=None, figsize=(8, 6), save=True,
                                      ftype=args.img_file_type, title_suffix=suffix,
                                      save_id_suffix=save_id, args=args,
                                      annotate_points=None)
            # Test
            encoder.to(torch.device('cpu'))
            classifier.to(torch.device('cpu'))
            model = get_net(args)
            state_dict = encoder.to(torch.device('cpu')).state_dict()
            model = load_encoder_state_dict(model, state_dict)
            try:
                model.fc = classifier
            except:
                model.classifier = classifier
                
            if epoch + 1 < args.max_epoch:
                evaluate_model(model, [train_loader, test_loader],
                               ['Training', 'Testing'],
                               test_criterion, args, epoch)
                
                print(f'Experiment name: {args.experiment_name}')
                
                
                if args.replicate in range(20, 30):
                    slice_outputs = compute_slice_outputs(model,
                                                          train_loader,
                                                          test_criterion,
                                                          args)
                    sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
                    contrastive_points = prepare_contrastive_points(sliced_data_indices,
                                                                    sliced_data_losses,
                                                                    sliced_data_correct,
                                                                    train_loader, args)
                    slice_anchors, slice_negatives, positives_by_class, all_targets = contrastive_points
                    
                if args.resample_class != '':
                    slice_outputs = recompute_slices_with_resampling(train_loader,
                                                                     slice_model,
                                                                     args.resample_class,
                                                                     test_criterion,
                                                                     seed=epoch+1+args.seed,
                                                                     args=args,
                                                                     split='Train')
                    sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
                    contrastive_points = prepare_contrastive_points(sliced_data_indices,
                                                                    sliced_data_losses,
                                                                    sliced_data_correct,
                                                                    train_loader, args)
                    slice_anchors, slice_negatives, positives_by_class, all_targets = contrastive_points

                contrastive_dataloader = load_contrastive_data(train_loader, 
                                                               slice_anchors, 
                                                               slice_negatives, 
                                                               positives_by_class,
                                                               epoch + 1 + args.seed, 
                                                               args)
            else:
                if args.finetune_epochs > 0:
                    dataloaders = (train_loader, val_loader, test_loader)
                    model = finetune_model(encoder, criterion,
                                           test_criterion, dataloaders,
                                           slice_model, args)
                
                args.model_type = 'final'
                run_final_evaluation(model, test_loader, test_criterion,
                                     args, epoch, visualize_representation=True)

                print('Done training')
                print(f'- Experiment name: {args.experiment_name}')
                print_header(f'Max Robust Acc:')
                print(f'Acc: {args.max_robust_acc}')
                print(f'Epoch: {args.max_robust_epoch}')
                summarize_acc(args.max_robust_group_acc[0],
                              args.max_robust_group_acc[1])
            
            
if __name__ == '__main__':
    main()