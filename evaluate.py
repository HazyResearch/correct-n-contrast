"""
Epoch evaluation functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from activations import visualize_activations
from network import save_checkpoint
from train import test_model
from utils.logging import summarize_acc
from utils.visualize import plot_data_batch, plot_confusion



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
