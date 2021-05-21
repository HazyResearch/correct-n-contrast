"""
Model architecture
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from collections import OrderedDict
# conda install -c huggingface transformers
from transformers import BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup


def get_net(args, pretrained=None):
    """
    Return model architecture
    """
    pretrained = args.pretrained if pretrained is None else pretrained
    if args.arch == "base":
        net = BaseNet(input_dim=args.d_causal + args.d_spurious,
                      hidden_dim_1=args.hidden_dim_1)
    elif args.arch == "logistic":
        net = LogisticRegression(input_dim=args.d_causal + args.d_spurious)
    elif 'mlp' in args.arch:
        net = MLP(num_classes=args.num_classes,
                  hidden_dim=args.hidden_dim)
        # net.activation_layer = 'relu'
    elif 'cnn' in args.arch:
        net = CNN(num_classes=args.num_classes)
    elif 'resnet' in args.arch:
        if 'resnet50' in args.arch:
            pretrained = True if '_pt' in args.arch else False
            net = torchvision.models.resnet50(pretrained=pretrained)
            d = net.fc.in_features
            net.fc = nn.Linear(d, args.num_classes)
        elif args.arch == 'resnet34':
            pretrained = True if '_pt' in args.arch else False
            net = torchvision.models.resnet34(pretrained=pretrained)
            d = net.fc.in_features
            net.fc = nn.Linear(d, args.num_classes)
        net.activation_layer = 'avgpool'
    elif 'densenet' in args.arch:
        pretrained = True if '_pt' in args.arch else False
        net = torchvision.models.densenet121(pretrained=pretrained)
        num_ftrs = net.classifier.in_features
        # add final layer with # outputs in same dimension of labels with sigmoid
        N_LABELS = 2  # originally 14 for pretrained model, but try this
        # activation
        net.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
        net.activation_layer = 'features.norm5'
    elif 'bert' in args.arch:
        if args.arch[-3:] == '_pt':
            model_name = args.arch[:-3]
        else:
            model_name = args.arch
            
        assert args.num_classes is not None
        assert args.task is not None
        
        config_class = BertConfig
        model_class = BertForSequenceClassification
        
        config = config_class.from_pretrained(model_name,
                                              num_labels=args.num_classes,
                                              finetuning_task=args.task)
        net = model_class.from_pretrained(model_name, from_tf=False, 
                                          config=config)
        # Either before or after the nonlinearity
        # net.activation_layer = 'bert.pooler.dense'
        net.activation_layer = 'bert.pooler.activation'
        # print(f'net.activation_layer: {net.activation_layer}')
    else:
        raise NotImplementedError
    return net


def get_output(model, inputs, labels, args):
    """
    General method for BERT and non-BERT model inference
    - Model and data batch should be passed already
    
    Args:
    - model (torch.nn.Module): Pytorch network
    - inputs (torch.tensor): Data features batch
    - labels (torch.tensor): Data labels batch
    - args (argparse): Experiment args
    """
    if args.arch == 'bert-base-uncased_pt':
        input_ids   = inputs[:, :, 0]
        input_masks = inputs[:, :, 1]
        segment_ids = inputs[:, :, 2]
        outputs = model(input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=labels)
        if labels is None:
            return outputs.logits
        return outputs[1]  # [1] returns logits
        # passing this into cross_entropy_loss gets a different loss
#     elif 'bert' in args.arch:
#         input_ids   = inputs[:, :, 0]
#         input_masks = inputs[:, :, 1]
#         segment_ids = inputs[:, :, 2]
#         outputs = model(input_ids=input_ids,
#                         attention_mask=input_masks,
#                         token_type_ids=segment_ids,
#                         labels=labels)
#         return outputs[1]
    else:
        return model(inputs)


def backprop_(model, optimizer, train_stage, args, scheduler=None):
    """
    General method for BERT and non-BERT backpropogation step
    - loss.backward() should already be called
    
    Args:
    - model (torch.nn.Module): Pytorch network
    - optimizer (torch.optim): Pytorch network's optimizer
    - train_stage (str): Either 'spurious', 'contrastive', 'grad_align'
    - args (argparse): Experiment args
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    """
    if train_stage == 'grad_align':
        clip_grad_norm = args.grad_clip_grad_norm
    else:
        clip_grad_norm = args.clip_grad_norm
        
    if args.arch == 'bert-base-uncased_pt' and args.optim == 'AdamW':
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()
    else:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    

def load_pretrained_model(path, args):
    checkpoint = torch.load(path)
    net = get_net(args)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    return net


def save_checkpoint(model, optim, loss, epoch, batch, args,
                    replace=True, retrain_epoch=None,
                    identifier=None):
    optim_state_dict = optim.state_dict() if optim is not None else None
    save_dict = {'epoch': epoch,
                 'batch': batch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optim_state_dict,
                 'loss': loss}
    if retrain_epoch is not None:
        epoch = f'{epoch}-cpre={retrain_epoch}'
    cpb_str = f'-cpb={batch}' if batch is not None else ''
    fname = f'cp-{args.experiment_name}-cpe={epoch}{cpb_str}.pth.tar'
    
    if identifier is not None:
        fname = fname.replace('cp-', f'cp-{identifier}-')
    fpath = os.path.join(args.model_path, fname)
    
    print(f'replace: {replace}')
    if replace is True:
        for f in os.listdir(args.model_path):
            if f.split('-cpe=')[0] == fname.split('-cpe=')[0]:
                # This one may not be necessary
                if (f.split('-cpe=')[-1].split('-')[0] != str(epoch) or 
                    f.split('-cpb=')[-1].split('.')[0] != str(batch)):
                    print(f'-> Updating checkpoint {f}...')
                    os.remove(os.path.join(args.model_path, f))
    if args.dataset == 'isic':
        fpaths = fpath.split('-r=210')
        fpath = fpaths[0] + fpaths[-1]
    try:
        torch.save(save_dict, fpath)
        print(f'Checkpoint saved at {fpath}')
    except:
        torch.save(save_dict, fname)
        print(f'Checkpoint saved at {fname}')
    del save_dict
    return fname


def get_optim(net, args, model_type='pretrain', 
              scheduler_lr=None):
    if model_type == 'spurious':
        lr = args.lr_s
        momentum = args.momentum_s
        weight_decay = args.weight_decay_s
        adam_epsilon = args.adam_epsilon_s
    elif model_type == 'classifier':
        # Repurposed for classifier
        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay_c
        adam_epsilon = args.adam_epsilon
    else:
        lr = args.lr if scheduler_lr is None else scheduler_lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        adam_epsilon = args.adam_epsilon
        
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
        
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999), 
                               eps=1e-08,
                               weight_decay=weight_decay,
                               amsgrad=False)

    elif args.optim == 'AdamW':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in net.named_parameters() 
                        if not any(nd in n for nd in no_decay)], 
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in net.named_parameters() 
                        if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}]
        optimizer = optim.AdamW(optimizer_grouped_parameters,
                                lr=lr, eps=adam_epsilon)
    else:
        raise NotImplementedError
    return optimizer


def get_bert_scheduler(optimizer, n_epochs, warmup_steps, dataloader, last_epoch=-1):
    """
    Learning rate scheduler for BERT model training
    """
    num_training_steps = int(np.round(len(dataloader) * n_epochs))
    print(f'\nt_total is {num_training_steps}\n')
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                warmup_steps,
                                                num_training_steps,
                                                last_epoch)
    return scheduler

# From pytorch-transformers:
def _get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                     num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_criterion(args, reduction='mean'):
    if args.criterion == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction=reduction)
    else:
        raise NotImplementedError


class BaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=20):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, 2)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

    def predict(self, x):
        x = F.softmax(self.forward(x))
        return np.argmax(x.detach().numpy())

    def embed(self, x, relu=False):
        x = self.fc1(x)
        return F.relu(self.fc2(x)) if relu else self.fc2(x)

    def last_layer_output(self, x, relu=False):
        """
        Opposite to embed, return softmax based on last layer
        Args:
        - x (torch.tensor): neural network embeddings
        - relu (bool): Whether x has ReLU applied to it
        Output:
        - Neural network output given hidden layer representation
        """
        return self.fc(x) if relu else self.fc(F.relu(x))


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc1(x)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5
        self.fc2 = nn.Linear(120, 84)  # Activations layer
        self.fc = nn.Linear(84, num_classes)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        
        self.activation_layer = torch.nn.ReLU

    def forward(self, x):
        # Doing this way because only want to save activations
        # for fc linear layers - see later
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x))
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        
        self.activation_layer = torch.nn.ReLU

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x))
        x = self.fc(x)
        return x
