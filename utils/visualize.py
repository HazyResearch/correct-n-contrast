"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import umap

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from sklearn.manifold import MDS

from os.path import join
# from train import get_embeddings


def plot_data_batch(dataset, mean=0.0, std=1.0, nrow=8, title=None,
                    args=None, save=False, save_id=None, ftype='png'):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except:
        print(f'Nothing to plot!')
        return
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    if save:
        try:
            fpath = join(args.image_path,
                         f'{save_id}-{args.experiment_name}.{ftype}')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        except Exception as e:
            fpath = f'{save_id}-{args.experiment_name}.{ftype}'
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
    if args.display_image:
        plt.show()
    plt.close()


def visualize_dataset(dataset, alpha=0.5):
    """
    Visualize dataset with 
    """
    all_data = dataset.data
    all_labels = dataset.targets_all['causal_t']
    all_labels_s = dataset.targets_all['spurious_t']
    plot_2d_toy_data(all_data, y_c=all_labels, y_s=all_labels_s, alpha=alpha)


def plot_2d_toy_data(X, y_c, y_s, title=None, cmap='RdBu', alpha=0.5):
    # Fancier version of plt.scatter(x=data[:, 0], y=data[:, 1], c=targets_c, cmap='RdBu', alpha=0.5)
    for c in np.unique(y_c):
        row_ix_c = np.where(y_c == c)
        for c_ in np.unique(y_s):
            row_ix_s = np.where(y_s == c_)
            if c == c_:
                edgecolor = 'black'
                group = 'maj'
                marker = '.'
            else:
                edgecolor = 'black'
                group = 'min'
                marker = '.'
            combined_row_ix = (np.intersect1d(row_ix_c[0], row_ix_s[0]), )
            colors = y_c[combined_row_ix] / 2 + 0.25
            colors = 'red' if np.unique(colors) > 0.5 else 'blue'
            plt.scatter(X[combined_row_ix, 0], X[combined_row_ix, 1],
                        c=colors, alpha=alpha, label=f'Causal: {c}, Spurious: {c_}',
                        marker=marker)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.075), fancybox=False, shadow=False, ncol=2)


def get_softmax(x, net, embeddings=False, relu=False):
    """
    Retrieve softmax output of neural network
    Args:
    - x (torch.tensor): Input tensor; input features by default, but should be encoded representation if embeddings=True
    - net (torch.nn.Module): Neural network
    - embeddings (bool): If true, get softmax based on neural net embeddings
    - relu (bool): Whether embeddings have ReLU function applied to them or not
    """
    dims = 1 if len(x.shape) == 2 else 0
    with torch.no_grad():
        x = torch.from_numpy(x).type(torch.FloatTensor)
        # print(f'torch.from_numpy(x).type(torch.FloatTensor).shape: {x.shape}')
        if embeddings:
            output = F.softmax(net.last_layer_output(x), dim=dims)
        else:
            output = F.softmax(net(x), dim=dims)
    return output


def plot_decision_boundary(net, X, y, y_spur, plot_embeddings=False,
                           relu=False, h=0.01, alpha=0.5, imshow=True,
                           title=None, args=None, save_id=None, save=True,
                           ftype='png'):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    embeddings = []

    # logits = get_softmax(X, net).detach().cpu().numpy()
    logits = get_softmax(grid, net, plot_embeddings,
                         relu).detach().cpu().numpy()
    # Just keep the probabilities for class 0
    logits = logits[:, 0]
    zz = logits.reshape(xx.shape)
    left_bottom_right_top = np.concatenate([grid[0], grid[-1]])

    if imshow:
        c = plt.imshow(zz, extent=left_bottom_right_top[[0, 2, 1, 3]],
                       cmap=plt.cm.RdBu, origin='lower', aspect='auto')
    else:
        c = plt.contourf(xx, yy, zz, cmap=plt.cm.RdBu)
    plt.colorbar(c)
    for c in np.unique(y):
        row_ix_c = np.where(y == c)
        for c_ in np.unique(y_spur):
            row_ix_s = np.where(y_spur == c_)
            if c == c_:
                edgecolor = 'black'
                group = 'maj'
                marker = 'o'
            else:
                edgecolor = 'white'
                group = 'min'
                marker = 'o'
            combined_row_ix = (np.intersect1d(row_ix_c[0], row_ix_s[0]), )
            colors = y[combined_row_ix] / 2 + 0.25
            colors = 'red' if np.unique(colors) > 0.5 else 'blue'
            plt.scatter(X[combined_row_ix, 0], X[combined_row_ix, 1],
                        c=colors, edgecolor=edgecolor, alpha=alpha,
                        label=f'Causal: {c}, Spurious: {c_}', marker=marker)
    if title is not None:
        plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
               fancybox=False, shadow=False, ncol=2)
    if save:
        fpath = join(args.image_path,
                     f'db-{save_id}-{args.experiment_name}.{ftype}')
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        print(f'Saved decision boundary visualization to {fpath}!')
    if args.display_image:
        plt.show()
    plt.close()


def plot_group_bars(dataset, alpha=0.75, title=None,
                    args=None, save_id=None, save=True,
                    ftype='png'):
    groups = dataset.group_labels
    y_pos = np.arange(len(groups))
    counts = np.zeros(len(groups))
    for ix in range(len(counts)):
        counts[ix] += len(np.where(dataset.groups == ix)[0])

    plt.bar(y_pos, counts, align='center', alpha=alpha)

    for ix, count in enumerate(counts):
        print(f'Group: {groups[ix]}: {count}')
    plt.xticks(y_pos, groups)
    plt.ylabel('Counts')
    plt.xlabel('Groups')
    if save:
        fpath = join(args.image_path,
                     f'gb-{save_id}-{args.experiment_name}.{ftype}')
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        print(f'Saved bar graph of groups to {fpath}!')
    if args.display_image:
        plt.show()
    plt.close()


def plot_test_decision_boundaries(net, test_loader, features=True,
                                  embeddings=True, activations=True,
                                  save_id_prefix='', args=None, save=True):
    """
    Plot and save all specified decision boundaries
    Args:
    - test_loader (torch.utils.data.DataLoader): Test set dataloader
    - net (torch.nn.Module): Trained network
    - features (bool): If true, plot decision boundary on input features
    - embeddings (bool): If true, plot on embeddings
    - activations (bool): If true, plot on activations
    """
    net.to(torch.device('cpu'))
    test_data = test_loader.dataset
    if features and args.d_causal + args.d_spurious <= 2:
        plot_decision_boundary(net, test_data.data,
                               test_data.targets_all['causal_t'],
                               test_data.targets_all['spurious_t'],
                               plot_embeddings=False, relu=False,
                               title=f'(Input Features), Train batch size: {args.bs_trn}, P Corr.: {args.p_correlation}, Causal Var.: {args.var_causal}, Spurious Var.: {args.var_spurious}',
                               args=args,
                               save_id=f'{save_id_prefix}-input',
                               save=save,
                               ftype=args.img_file_type)
    if embeddings or activations:
        test_embeddings, test_embeddings_r = get_embeddings(
            net, test_loader, args)

    if embeddings:
        net.to(torch.device('cpu'))
        plot_decision_boundary(net, test_embeddings,
                               test_data.targets_all['causal_t'],
                               test_data.targets_all['spurious_t'],
                               plot_embeddings=True, relu=True,
                               title=f'(Embeddings), Train batch size: {args.bs_trn}, P Corr.: {args.p_correlation}, Causal Var.: {args.var_causal}, Spurious Var.: {args.var_spurious}',
                               args=args,
                               save_id=f'{save_id_prefix}-embed',
                               save=save,
                               ftype=args.img_file_type)
    if activations:
        net.to(torch.device('cpu'))
        plot_decision_boundary(net, test_embeddings_r,
                               test_data.targets_all['causal_t'],
                               test_data.targets_all['spurious_t'],
                               plot_embeddings=True, relu=True,
                               title=f'(Embeddings ReLU), Train batch size: {args.bs_trn}, P Corr.: {args.p_correlation}, Causal Var.: {args.var_causal}, Spurious Var.: {args.var_spurious}',
                               args=args,
                               save_id=f'{save_id_prefix}-relu',
                               save=save,
                               ftype=args.img_file_type)


def plot_umap(embeddings, dataset, label_type, num_data=None, method='umap',
              offset=0, figsize=(12, 9), save_id=None, save=True,
              ftype='png', title_suffix=None, args=None, cmap='tab10',
              annotate_points=None, predictions=None):
    """
    Visualize embeddings with U-MAP
    """
    labels = predictions if label_type == 'predictions' else dataset.targets_all[label_type]
    if num_data is None:
        embeddings = embeddings
    elif offset == 0:
#         sample_ix = np.arange(0, len(embeddings), 
#                               int(len(embeddings) / num_data))
        np.random.seed(args.seed)
        num_data = np.min((num_data, len(embeddings)))
        sample_ix = np.random.choice(np.arange(len(embeddings)),
                                     size=num_data, replace=False)
        embeddings = embeddings[sample_ix]
        labels = labels[sample_ix]
    else:
        embeddings = embeddings[offset:offset + num_data]
        labels = labels[offset:offset + num_data]
        
    if method == 'umap':
        standard_embedding = umap.UMAP(random_state=42).fit_transform(embeddings)
    else:  # method == 'mds'
        standard_embedding = MDS(n_components=2,
                                 random_state=42).fit_transform(embeddings)
    colors = np.array(labels).astype(int)
    num_colors = len(np.unique(colors))
    plt.figure(figsize=figsize)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1],
                c=colors, s=1.0, alpha=1,
                cmap=plt.cm.get_cmap(cmap, num_colors))  # 'tab10', 'set1', 'gist_rainbow'
    if annotate_points is not None:
        for i, txt in enumerate(range(len(standard_embedding[:, 0]))):
            try:
                if i % annotate_points == 0 and dataset.targets_all['group_idx'][i] in [0, 2]:  # For
                    color = plt.cm.get_cmap(cmap, num_colors).colors[
                        dataset.targets_all['group_idx'][i]]
                    erm_pred = dataset.targets_all['erm_pred'][i]
                    plt.annotate(erm_pred, (standard_embedding[:, 0][i],
                                            standard_embedding[:, 1][i]),
                                 fontsize=8,
                                 color=tuple(color))
            except:
                # Only annotate the group_idx UMAP
                pass # print(plt.cm.get_cmap(cmap, num_colors).colors)
    suffix = '' if title_suffix is None else f' {title_suffix}'
    plt.title(f'Color by {label_type} labels{suffix}')
    plt.colorbar(ticks=np.unique(colors))
    if save:
        try:
            fpath = join(args.image_path,
                         f'{method}-{save_id}-{args.experiment_name}.{ftype}')
            fpath = fpath.replace('..', '.')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
            print(f'Saved {method} to {fpath}!')
        except Exception as e:
            print(e)
            fpath = f'{method}-{save_id}-{args.experiment_name}.{ftype}'
            fpath = fpath.replace('1e-05', '1e_5')
            fpath = fpath.replace('0.00001', '1e_5')
            fpath = fpath.replace('1e-04', '1e_4')
            fpath = fpath.replace('0.0001', '1e_4')
            if args.dataset == 'isic':
                fpaths = fpath.split('-r=210')
                fpath = fpaths[0] + fpaths[-1]
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
            print(f'Saved {method} to {fpath}!')
    if args.display_image:
        plt.show()
    plt.close('all')
    del standard_embedding
    


def plot_confusion(correct_by_groups, total_by_groups, save_id=None, save=True,
                   ftype='png', args=None):
    matrix = correct_by_groups / total_by_groups
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    targets = (np.arange(correct_by_groups.shape[0]) + 1).astype(int)
    spurious = (np.arange(correct_by_groups.shape[0]) + 1).astype(int)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(targets)))
    ax.set_yticks(np.arange(len(spurious)))
    ax.set_xticklabels(targets)
    ax.set_yticklabels(spurious)
    ax.set_xlabel = 'Target'
    ax.set_ylabel = 'Spurious'

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right")

    ax.figure.colorbar(im, ax=ax)

    ax.set_title(f"Target / Spurious Accuracies ({save_id})")
    fig.tight_layout()
    if save:
        fpath = join(args.image_path,
                     f'cm-{save_id}-{args.experiment_name}.{ftype}')
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        # print(f'Saved bar graph of groups to {fpath}!')
    if args.display_image:
        plt.show()
    plt.close()


def plot_misclassified_bars(indices, all_groups, labels):
    groups, counts = np.unique(all_groups[indices], return_counts=True)
    x_pos = np.arange(len(groups))
    plt.bar(x_pos, counts, align='center')
    plt.xticks(x_pos, labels)
    plt.title('Incorrect classifications')
    plt.show()
