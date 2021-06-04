"""
Correct-n-Contrast main script
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
from datasets import train_val_split, get_resampled_indices, get_resampled_set, initialize_data
# Logging and training
from train import train_model, test_model
from evaluate import evaluate_model, run_final_evaluation
from utils import free_gpu, print_header  # , update_contrastive_experiment_name
from utils import init_experiment, init_args, update_args
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
from contrastive_network import ContrastiveNet, load_encoder_state_dict, compute_outputs
from contrastive_network import SupervisedContrastiveLoss
from slice import compute_pseudolabels, compute_slice_indices, train_spurious_model
## Alternative slicing by UMAP clustering
from slice_rep import compute_slice_indices_by_rep, combine_data_indices

import transformers
transformers.logging.set_verbosity_error()





def train_epoch(encoder, classifier, dataloader,
                optim_e, optim_c, scheduler_e, scheduler_c,
                epoch, val_loader, contrastive_loss,
                cross_entropy_loss, args):
    """
    Train contrastive epoch
    """
    encoder.to(args.device)
    classifier.to(args.device)
    
    optim_e.zero_grad()
    optim_c.zero_grad()
    contrastive_weight = args.contrastive_weight
    loss_compute_size = int(args.num_anchor +
                            args.num_negative +
                            args.num_positive +
                            args.num_negative_easy)
    epoch_losses = []
    epoch_losses_contrastive = []
    epoch_losses_cross_entropy = []
    
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
                loss.backward()
                contrastive_batch = contrastive_batch.detach().cpu()
                
                batch_loss += loss.item()
                batch_loss_contrastive += loss.item()
                free_gpu([loss], delete=True)
                
                # Two-sided contrastive update
                if args.num_negative_easy > 0:
                    contrastive_batch = torch.vstack(
                        (inputs_p[0].unsqueeze(0), inputs_a, inputs_ne)
                    )
                    # Compute contrastive loss
                    loss = contrastive_loss(encoder, contrastive_batch)
                    loss *= ((1 - supervised_weight) / 
                             (len(inputs_a_) * len(batch_inputs)))
                    

                    loss.backward()
                    contrastive_batch = contrastive_batch.detach().cpu()

                    batch_loss += loss.item()
                    batch_loss_contrastive += loss.item()
                    free_gpu([loss], delete=True)
                    
                if args.finetune_epochs > 0:
                    continue
                
                # Compute cross-entropy loss jointly
                if anchor_ix + 1 == len(inputs_a_):
                    input_list = [inputs_a, inputs_p, inputs_n, inputs_ne]
                    label_list = [labels_a, labels_p, labels_n, labels_ne]
                    min_input_size = np.min([len(x) for x in input_list])
                    contrast_inputs = torch.cat([x[:min_input_size] for x in input_list])
                    contrast_labels = torch.cat([l[:min_input_size] for l in label_list])
                    if loss_compute_size <= args.bs_trn:
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
                        shuffle_ix = np.arange(contrast_inputs.shape[0])
                        np.random.shuffle(shuffle_ix)
                        contrast_inputs = contrast_inputs[shuffle_ix]
                        contrast_labels = contrast_labels[shuffle_ix]
                        
                        contrast_inputs = torch.split(contrast_inputs,
                                                      args.bs_trn)
                        contrast_labels = torch.split(contrast_labels,
                                                      args.bs_trn)

                        for cix, contrast_input in enumerate(contrast_inputs):
                            weight = contrast_input.shape[0] / len(shuffle_ix)
                            output, loss = compute_outputs(contrast_input, 
                                                           encoder,
                                                           classifier,
                                                           args,
                                                           contrast_labels[cix], 
                                                           True,
                                                           cross_entropy_loss)
                            loss *= (supervised_weight * weight /
                                     len(batch_inputs))
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

            if args.replicate > 50:
                optim_c.zero_grad()
        
        epoch_losses.append(batch_loss)
        epoch_losses_contrastive.append(batch_loss_contrastive)
        epoch_losses_cross_entropy.append(batch_loss_cross_entropy)
        
        if (batch_ix + 1) % args.log_loss_interval == 0:
            print_output  = f'Epoch {epoch:>3d} | Batch {batch_ix:>4d} | '
            print_output += f'Loss: {batch_loss:<.4f} (Epoch Avg: {np.mean(epoch_losses):<.4f}) | '
            print_output += f'CL: {batch_loss_contrastive:<.4f} (Epoch Avg: {np.mean(epoch_losses_contrastive):<.4f}) | '
            print_output += f'CE: {batch_loss_cross_entropy:<.4f}, (Epoch Avg: {np.mean(epoch_losses_cross_entropy):<.4f}) | '
            print_output += f'SW: {supervised_weight:<.4f}'
            print(print_output)
            
        if ((batch_ix + 1) % args.checkpoint_interval == 0 or 
            (batch_ix + 1) == len(dataloader)):
            model = get_net(args)
            state_dict = encoder.to(torch.device('cpu')).state_dict()
            model = load_encoder_state_dict(model, state_dict)
            if 'bert' in args.arch:
                model.classifier = classifier
            else:
                model.fc = classifier
            checkpoint_name = save_checkpoint(model, None,
                                              np.mean(epoch_losses),
                                              epoch, batch_ix, args,
                                              replace=True,
                                              retrain_epoch=-1,
                                              identifier='fm')
            args.checkpoint_name = checkpoint_name
            
    epoch_losses = (epoch_losses,
                    epoch_losses_contrastive,
                    epoch_losses_cross_entropy)
    return encoder, classifier, epoch_losses
        
        
def compute_slice_outputs(erm_model, train_loader, test_criterion, args):
    """
    Compute predictions of ERM model to set up contrastive batches
    """
    if 'rep' in args.slice_with:
        slice_outputs = compute_slice_indices_by_rep(erm_model,
                                                     train_loader,
                                                     cluster_umap=True, 
                                                     umap_components=2,
                                                     cluster_method=args.rep_cluster_method,
                                                     args=args,
                                                     visualize=True)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs

    if 'pred' in args.slice_with:
        slice_outputs_ = compute_slice_indices(erm_model, train_loader, 
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
    elif args.slice_with == 'pred':
        sliced_data_indices = sliced_data_indices_
        sliced_data_correct = sliced_data_correct_
        sliced_data_losses = sliced_data_losses_
        
    return sliced_data_indices, sliced_data_correct, sliced_data_losses


def finetune_model(encoder, criterion, test_criterion, dataloaders, 
                   erm_model, args):
    """
    Instead of joint training, finetune classifier
    """
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
    erm_model.to(args.device)
    erm_model.eval()
    slice_outputs = compute_slice_outputs(erm_model,
                                          train_loader,
                                          test_criterion, 
                                          args)
    sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
    erm_model.to(torch.device('cpu'))
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
    model, max_robust_metrics, all_acc = outputs
    return model
        

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
    # update_contrastive_experiment_name(args)
    update_args(args)
    
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
        

    if args.evaluate is True:
        initialize_csv_metrics(args)
        assert args.load_encoder != ''
        args.checkpoint_name = args.load_encoder
        try:
            start_epoch = int(args.checkpoint_name.split('-cpe=')[-1].split('-')[0])
        except:
            start_epoch = 0
        try:  # Load full model
            print(f'Loading full model...')
            model = get_net(args)
            model_state_dict = torch.load(os.path.join(args.model_path,
                                                       args.checkpoint_name))
            model_state_dict = model_state_dict['model_state_dict']
            model = load_encoder_state_dict(model, model_state_dict,
                                            contrastive_train=False)
            print(f'-> Full model loaded!')
        except Exception as e:
            print(e)
            project = not args.no_projection_head
            assert args.load_encoder != ''
            args.checkpoint_name = args.load_encoder
            start_epoch = int(args.checkpoint_name.split('-cpe=')[-1].split('-')[0])
            checkpoint = torch.load(os.path.join(args.model_path,
                                                 args.checkpoint_name))
            print(f'Checkpoint loading from {args.load_encoder}!')
            print(f'- Resuming training at epoch {start_epoch}')

            
            encoder = ContrastiveNet(args.arch, out_dim=args.projection_dim, 
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
        erm_model = load_pretrained_model(args.pretrained_spurious_path, args)
        erm_model.eval()
        args.mode = 'train_spurious'
    else:
        args.mode = 'train_spurious'
        print_header('> Training spurious model')
        args.spurious_train_split = 0.99
        erm_model, outputs, _ = train_spurious_model(train_loader, args)

    erm_model.eval()
    print(f'Pretrained model loaded from {args.pretrained_spurious_path}')
    
    if args.train_encoder is True:
        slice_outputs = compute_slice_outputs(erm_model, train_loader,
                                              test_criterion, args)
        sliced_data_indices, sliced_data_correct, sliced_data_losses = slice_outputs
        
        for _, p in erm_model.named_parameters():
            p = p.to(torch.device('cpu'))
        erm_model.to(torch.device('cpu'))
        
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
        
        encoder = ContrastiveNet(args.arch, out_dim=args.projection_dim, 
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
        contrastive_loss = SupervisedContrastiveLoss(args)
        
        args.epoch_mean_loss = 1e5
        all_losses = []
        all_losses_cl = []
        all_losses_ce = []

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
                                        epoch, val_loader, 
                                        contrastive_loss, cross_entropy_loss,
                                        args)

            encoder, classifier, epoch_losses = train_outputs
            epoch_loss, epoch_loss_cl, epoch_loss_ce = epoch_losses
            all_losses.extend(epoch_loss)
            all_losses_cl.extend(epoch_loss_cl)
            all_losses_ce.extend(epoch_loss_ce)

            if 'bert' not in args.arch:
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
                visualize_activations(encoder, dataloader=val_loader,
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
                evaluate_model(model, [train_loader, val_loader],
                               ['Training', 'Validation'],
                               test_criterion, args, epoch)
                
                print(f'Experiment name: {args.experiment_name}')
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
                                           erm_model, args)
                
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