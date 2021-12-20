"""
Methods for sampling datapoints to organize and load contrastive datapoints

Methods:
- prepare_contrastive_points()
- sample_anchors()
- sample_positives()
- sample_negatives()
- load_contrastive_data()
"""

import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from datasets import get_resampled_set


def prepare_contrastive_points(sliced_data_indices,
                               sliced_data_losses,
                               sliced_data_correct,
                               train_loader, args):

    train_targets_all = train_loader.dataset.targets_all
    train_targets = train_targets_all['target']
    train_spurious = train_targets_all['spurious']
    sliced_data_indices_all = np.concatenate(sliced_data_indices)
    sliced_data_losses_all = np.zeros(len(train_targets))
    sliced_data_losses_all[sliced_data_indices_all] = np.concatenate(
        sliced_data_losses)
    sliced_data_correct_all = np.zeros(len(train_targets))
    sliced_data_correct_all[sliced_data_indices_all] = np.concatenate(
        sliced_data_correct)

    all_anchors = {'slice_ix': np.zeros(len(train_targets)).astype(int),
                   'in_slice_ix': np.zeros(len(train_targets)).astype(int)}

    # Store all anchors and negatives
    slice_anchors = [None] * len(sliced_data_indices)
    slice_negatives = [None] * len(sliced_data_indices)
    # Additional negatives, if specified
    additional_slice_negatives = [None] * len(sliced_data_indices)

    # For positives, just specify by the ground-truth
    # (These are the same as negatives in another slice, just organized by class)
    positives_by_class = {}

    for slice_ix, data_indices in enumerate(sliced_data_indices):
        target_class, target_counts = np.unique(train_targets[data_indices],
                                                return_counts=True)

        for tc_ix, tc in enumerate(target_class):
            print(
                f'>> Slice {slice_ix}, target: {tc}, counts: {target_counts[tc_ix]}')

        # Anchors are datapoints in the slice that the model got correct
        ix = np.where(sliced_data_correct[slice_ix])[0]
        print(
            f'Slice {slice_ix} % correct: {len(ix) / len(data_indices) * 100:<.4f} %')

        slice_ix_anchors = {'ix': data_indices[ix],
                            'loss': sliced_data_losses[slice_ix][ix],
                            'target': train_targets[data_indices][ix],
                            'correct': sliced_data_correct[slice_ix][ix],
                            'source': np.ones(len(data_indices[ix])).astype(int) * slice_ix,
                            'spurious': train_spurious[data_indices][ix],
                            'ix_by_class': {},
                            'loss_by_class': {}}

        for t in np.unique(train_targets[data_indices][ix]):
            tix = np.where(train_targets[data_indices][ix] == t)[0]
            slice_ix_anchors['ix_by_class'][t] = data_indices[ix][tix]
            slice_ix_anchors['loss_by_class'][t] = sliced_data_losses[slice_ix][ix][tix]

        # Negatives, slice-specific. All incorrect datapoints in the slice
        nix = np.setdiff1d(np.arange(len(data_indices)), ix)
        # TODO: handle case if there are no incorrect datapoints
        if len(nix) == 0:
            avg_correct = []
            for c in np.unique(train_targets[data_indices]):
                class_indices = np.where(train_targets[data_indices] == c)[0]
                class_correct = sliced_data_correct[slice_ix][class_indices]
                # avg_correct.append(np.mean(class_correct))
                avg_correct.append(len(class_correct))
            max_class_ix = np.argmax(avg_correct)
            max_class = target_class[max_class_ix]
            neg_class_ix = np.where(train_targets != max_class)[0]
            slice_ix_negatives = {'ix': list(neg_class_ix),
                                  'loss': list(sliced_data_losses_all[neg_class_ix]),
                                  'target': list(train_targets[neg_class_ix]),
                                  # source not technically right here
                                  'correct': list(sliced_data_correct_all[neg_class_ix]),
                                  'source': list((np.ones(len(train_targets_all)) * slice_ix).astype(int)),
                                  'spurious': [None]}
        else:
            print(f'Slice {slice_ix} # negative (incorrect): {len(nix)}')
            print(
                f'Slice {slice_ix} % negative (incorrect): {len(nix) / len(data_indices) * 100 :<.4f} %')
            print(
                f'Unique negative targets: {np.unique(train_targets[data_indices][nix], return_counts=True)}')

            slice_ix_negatives = {'ix': list(data_indices[nix]),
                                  'loss': list(sliced_data_losses[slice_ix][nix]),
                                  'target': list(train_targets[data_indices][nix]),
                                  'correct': list(sliced_data_correct[slice_ix][nix]),
                                  'source': list(np.ones(len(data_indices[nix])).astype(int) * slice_ix),
                                  'spurious': list(train_spurious[data_indices][nix])}

            # Positives, for other slices - for here just save by unique class that was also incorrect
            target_class, target_counts = np.unique(train_targets[data_indices][nix],
                                                    return_counts=True)
            incorrect_data_indices = data_indices[nix]
            for cix, c in enumerate(target_class):
                pix = np.where(train_targets[incorrect_data_indices] == c)[0]
                pos_data_indices = list(incorrect_data_indices[pix])
                pos_data_losses = list(sliced_data_losses[slice_ix][nix][pix])
                pos_data_targets = list(
                    train_targets[incorrect_data_indices][pix])
                pos_data_correct = list(
                    sliced_data_correct[slice_ix][nix][pix])
                pos_data_source = list(
                    np.ones(len(data_indices[nix][pix])).astype(int) * slice_ix)
                pos_data_spurious = list(
                    train_spurious[incorrect_data_indices][pix])
                if c in positives_by_class:
                    positives_by_class[c]['ix'].extend(pos_data_indices)
                    positives_by_class[c]['loss'].extend(pos_data_losses)
                    positives_by_class[c]['target'].extend(pos_data_targets)
                    positives_by_class[c]['correct'].extend(pos_data_correct)
                    positives_by_class[c]['source'].extend(pos_data_source)
                    positives_by_class[c]['spurious'].extend(pos_data_spurious)
                else:
                    positives_by_class[c] = {'ix': pos_data_indices,
                                             'loss': pos_data_losses,
                                             'target': pos_data_targets,
                                             'correct': pos_data_correct,
                                             'source': pos_data_source,
                                             'spurious': pos_data_spurious}
        # Save
        slice_anchors[slice_ix] = slice_ix_anchors
        slice_negatives[slice_ix] = slice_ix_negatives

    # Fill in positives if no slices had the class as spurious
    for slice_ix, data_indices in enumerate(sliced_data_indices):
        target_class, target_counts = np.unique(train_targets[data_indices],
                                                return_counts=True)

        # Compare average correctness, still use the max_class variable
        avg_correct = []
        for c in target_class:
            class_indices = np.where(train_targets[data_indices] == c)[0]
            class_correct = sliced_data_correct[slice_ix][class_indices]
            avg_correct.append(np.mean(class_correct))
        max_class_ix = np.argmax(avg_correct)

        for c in target_class:
            if c not in positives_by_class:
                print(
                    f'> Loading correct datapoints as positives for class {c} from slice {slice_ix}')
                ix = np.where(train_targets[data_indices] == c)[0]
                positives_by_class[c] = {'ix': list(data_indices[ix]),
                                         'loss': list(sliced_data_losses[slice_ix][ix]),
                                         'target': list(train_targets[data_indices][ix]),
                                         'correct': list(sliced_data_correct[slice_ix][ix]),
                                         'source': list(np.ones(len(data_indices[ix])).astype(int) * slice_ix),
                                         'spurious': list(train_spurious[data_indices][ix])}

    # Convert casted lists back to ndarrays
    for c, class_dict in positives_by_class.items():
        for k, v in class_dict.items():
            positives_by_class[c][k] = np.array(v)

    for ix, slice_negative in enumerate(slice_negatives):
        for k, v in slice_negative.items():
            slice_negatives[ix][k] = np.array(v)

    return slice_anchors, slice_negatives, positives_by_class, all_anchors


def sample_anchors(anchor_class, anchor_dict,
                   num_anchor, weight_by_loss):
    p = None
    if weight_by_loss:
        exp = np.exp(anchor_dict['loss_by_class'][anchor_class] /
                     args.anc_loss_temp)
        p = exp / exp.sum()
    num_samples = num_anchor
    sample_indices = anchor_dict['ix_by_class'][anchor_class]
    replace = True if num_samples > len(sample_indices) else False
    sample_indices = np.random.choice(sample_indices,
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    return sample_indices


def sample_positives(anchor_class, positives_by_class,
                     num_positive, weight_by_loss):
    positive_dict = positives_by_class[anchor_class]
    p = None
    if weight_by_loss:  # Check this
        # Sample the ones with more loss more frequently
        exp = np.exp(positive_dict['loss'] / args.pos_loss_temp)
        p = exp / exp.sum()
    num_samples = num_positive
    replace = True if num_samples > len(positive_dict['ix']) else False

    sample_indices = np.random.choice(np.arange(len(positive_dict['ix'])),
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    sample_slice_sources = positive_dict['source'][sample_indices]
    sample_indices = positive_dict['ix'][sample_indices]
    return sample_indices, sample_slice_sources


def sample_negatives(negative_dict, num_negative,
                     weight_by_loss):
    p = None
    if weight_by_loss:
        exp = np.exp(negative_dict['loss'] / args.neg_loss_temp)
        p = exp / exp.sum()
    num_samples = num_negative
    replace = True if num_samples > len(negative_dict['ix']) else False
    sample_indices = np.random.choice(negative_dict['ix'],
                                      size=num_samples,
                                      replace=replace,
                                      p=p)
    return sample_indices


# Adjust number of negatives or positives if > sliced neg / pos
def adjust_num_pos_neg_(positives_by_class, slice_negatives,
                        args):
    num_pos = np.min([len(positives_by_class[c]['target'])
                      for c in range(args.num_classes)])
    num_neg = np.min([len(negative_dict['target'])
                      for negative_dict in slice_negatives])
    num_pos = np.min((args.num_positive, num_pos))
    num_neg = np.min((args.num_negative, num_neg))

    # Tentative
    num_anc = np.min((args.num_anchor, np.min((num_pos, num_neg))))

    # Adjust experiment name to reflect
    args.experiment_name = args.experiment_name.replace(
        f'-na={args.num_anchor}-np={args.num_positive}-nn={args.num_negative}',
        f'-na={num_anc}-np={num_pos}-nn={num_neg}')
    # Adjust arguments
    args.num_positive = num_pos
    args.num_negative = num_neg
    args.num_anchor = num_anc
    print(f'Adjusted number of anchors:   {args.num_anchor}')
    print(f'Adjusted number of positives: {args.num_positive}')
    print(f'Adjusted number of negatives: {args.num_negative}')


# Adjust number of anchors or hard negatives if > sliced anc / neg
def adjust_num_anc_neg_(slice_anchors, slice_negatives,
                        args):
    num_anc = np.min([len(anchor_dict['target'])
                      for anchor_dict in slice_anchors])
    num_neg = np.min([len(negative_dict['target'])
                      for negative_dict in slice_negatives])
    num_anc = np.min((args.num_anchor, num_anc))
    # num_neg Because now both anchors and negatives are from the nonspurious groups
    num_neg = np.min((args.num_negative_easy, num_anc))

    # Tentative
    # num_anc = np.min((args.num_anchor, np.min((num_pos, num_neg))))

    # Adjust experiment name to reflect
    args.experiment_name = args.experiment_name.replace(
        f'-na={args.num_anchor}-np={args.num_positive}-nn={args.num_negative}-ne={args.num_negative_easy}',
        f'-na={num_anc}-np={args.num_positive}-nn={args.num_negative}-ne={num_neg}')
    # Adjust arguments
    args.num_anchor = num_anc
    args.num_negative_easy = num_neg
    print(f'Adjusted number of anchors:   {args.num_anchor}')
    print(f'Adjusted number of (hard) negatives: {args.num_negative_easy}')


def load_contrastive_data(train_loader, slice_anchors,
                          slice_negatives, positives_by_class,
                          seed, args, supervised_contrast=True):
    # Get number of negatives per target class
    args.num_negatives_by_target = [0] * args.num_classes
    assert args.replicate % 2 == 0  # Checking / debugging

    batch_samples = []
    batch_samples_old = []
    if args.balance_targets:
        print(f'Debug: args.balance_targets: {args.balance_targets}')
        max_sample_size = np.min([len(anchor_dict['ix']) for anchor_dict in
                                  slice_anchors])

    for slice_ix, anchor_dict in enumerate(slice_anchors):
        batch_samples_per_slice = []  # First aggregate within
        negative_dict = slice_negatives[slice_ix]
        # For hard negative
        args.num_negatives_by_target[slice_ix] = len(negative_dict['ix'])

        if args.balance_targets:
            n_samples = int(np.round(args.target_sample_ratio *
                                     max_sample_size))
            print(f'slice {slice_ix} n_samples: {n_samples}')
            try:
                p = targets['p']
            except:
                p = None
            anchor_indices = np.random.choice(np.arange(len(anchor_dict['ix'])),
                                              size=n_samples,
                                              replace=False,
                                              p=p)
            anchor_targets = anchor_dict['target'][anchor_indices]
            anchor_indices = anchor_dict['ix'][anchor_indices]

        elif args.target_sample_ratio < 1:
            n_samples = int(np.round(args.target_sample_ratio *
                                     len(anchor_dict['ix'])))
            anchor_indices = np.random.choice(np.arange(len(anchor_dict['ix'])),
                                              size=n_samples,
                                              replace=False)
            anchor_targets = anchor_dict['target'][anchor_indices]
            anchor_indices = anchor_dict['ix'][anchor_indices]
        else:
            anchor_targets = anchor_dict['target']
            anchor_indices = anchor_dict['ix']

        for aix, anchor_ix in enumerate(tqdm(anchor_indices, desc=f'Generating data from slice {slice_ix}')):
            anchor_class = anchor_targets[aix]
            # Sample additional positives
            anchor_indices = sample_anchors(anchor_class,
                                            anchor_dict,
                                            args.num_anchor - 1,
                                            args.weight_anc_by_loss)
            anchor_indices = np.concatenate([[anchor_ix], anchor_indices])
            positive_outputs = sample_positives(anchor_class,
                                                positives_by_class,
                                                args.num_positive,
                                                args.weight_pos_by_loss)
            positive_indices, positive_slice_sources = positive_outputs
            # Keep as this, in case want to generate new neg per pos as before
            samples = [anchor_indices, positive_indices]
            negative_indices = sample_negatives(negative_dict,
                                                args.num_negative,
                                                args.weight_neg_by_loss)
            samples.append(negative_indices)
            # Sample second negatives ("easy kind")
            if args.num_negative_easy > 0:
                # Just sample from first one - for "easy negatives"
                anchor_slice = positive_slice_sources[0]
                negative_indices = sample_negatives(slice_anchors[anchor_slice],
                                                    args.num_negative_easy,
                                                    args.weight_neg_by_loss)
                samples.append(negative_indices)
            batch_sample = np.concatenate(samples)
            batch_samples_per_slice.append(batch_sample)
            batch_samples_old.append(batch_sample)
        np.random.shuffle(batch_samples_per_slice)
        batch_samples.append(batch_samples_per_slice)

    batch_samples = list(zip(*batch_samples))
    batch_samples = np.array(batch_samples).reshape(-1, len(batch_sample))

    contrastive_indices = np.concatenate(batch_samples)
    contrastive_train_set = get_resampled_set(train_loader.dataset,
                                              contrastive_indices,
                                              copy_dataset=True)

    contrastive_dataloader = DataLoader(contrastive_train_set,
                                        batch_size=len(
                                            batch_samples[0]) * int(args.batch_factor),
                                        shuffle=False, num_workers=args.num_workers)

    return contrastive_dataloader
