"""
Contrastive network architecture, loss, and functions
"""

import torch
import torch.nn as nn
import torchvision.models as models

from copy import deepcopy
from transformers import BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import free_gpu
from network import CNN, MLP, get_output  

from resnet import *


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
                elif (k.startswith('backbone.fc') or
                      k.startswith('backbone.classifier')):
                    pass
                else:
                    state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
    print(f'log.missing_keys: {log.missing_keys}')
    return model
    
    
class ContrastiveNet(nn.Module):

    def __init__(self, base_model, out_dim, projection_head=True,
                 task=None, num_classes=None, checkpoint=None):
        super(ContrastiveNet, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        
        if base_model[-3:] == '_pt':
            self.pretrained = True
            base_model = base_model[:-3]
        else:
            self.pretrained = False
        print(f'Loading with {base_model} backbone')
        self.base_model = base_model
        # Also adds classifier, retreivable with self.classifier
        self.backbone = self.init_basemodel(base_model)
        self.projection_head = projection_head
        self.backbone = self.init_projection_head(self.backbone, 
                                                  out_dim,
                                                  project=projection_head)
        
    def init_basemodel(self, model_name):
        try:
            if 'resnet50' in model_name:
                # model_name = 'resnet50'
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
                
            elif 'bert' in model_name:
                # model_name = 'bert-base-uncased'
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
        
        
    def init_projection_head(self, backbone, out_dim, project=True):
        if 'resnet' in self.base_model or 'cnn' in self.base_model or 'mlp' in self.base_model:
            dim_mlp = backbone.fc.in_features
            
            self.classifier = nn.Linear(dim_mlp, self.num_classes)
            if project:
                # Modify classifier head to match projection output dimension
                backbone.fc = nn.Linear(dim_mlp, out_dim)
                # Add projection head
                backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                                            nn.ReLU(), 
                                            backbone.fc)
            else:
                backbone.fc = nn.Identity(dim_mlp, -1)
            
        elif 'bert' in self.base_model:
            print(backbone)
            dim_mlp = backbone.classifier.in_features
            
            self.classifier = deepcopy(backbone.classifier)
            print(self.classifier)
            if project:
                backbone.classifier = nn.Linear(dim_mlp, out_dim)
                backbone.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    backbone.classifier)
            else:
                backbone.classifier = nn.Identity(dim_mlp, -1)
                print(backbone.classifier)
        self.dim_mlp = dim_mlp
        return backbone
    
    def forward(self, x):
        if self.base_model == 'bert-base-uncased':
            input_ids, input_masks, segment_ids, labels = x
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=input_masks,
                                    token_type_ids=segment_ids,
                                    labels=labels)
            if labels is None:
                return outputs.logits
            return outputs[1]  # [1] returns logits
        return self.backbone(x)
    
    def encode(self, x):
        if self.base_model == 'bert-base-uncased':
            input_ids   = x[:, :, 0]
            input_masks = x[:, :, 1]
            segment_ids = x[:, :, 2]
            x = (input_ids, input_masks, segment_ids, None)
        
        if self.projection_head:
            encoder = deepcopy(self.backbone)
            encoder.fc = nn.Identity(self.dim_mlp, -1)
            if self.base_model == 'bert-base-uncased':
                input_ids, input_masks, segment_ids, labels = x
                return encoder(input_ids=input_ids,
                               attention_mask=input_masks,
                               token_type_ids=segment_ids,
                               labels=labels)
            return encoder(x)
        else:
            return self.forward(x)
    

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = args.temperature
        self.n_positives = args.num_positive
        self.n_negatives = args.num_negative
        self.arch = args.arch
        self.args = args
        self.hard_neg_factor = args.hard_negative_factor
        try:
            self.single_pos = args.single_pos
        except:
            self.single_pos = False
        
        self.sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, model, contrastive_batch):
        # Compute negative similarities
        neg_indices = [0] + list(range(len(contrastive_batch))[
            -self.n_negatives:])
        anchor_negatives = contrastive_batch[neg_indices]
        exp_neg = self.compute_exp_sim(model, anchor_negatives,
                                       return_sum=False)
        # Hard negative reweighting - by default ignore
        if self.hard_neg_factor > 0:
            # exp_neg.mean() because N * E[... exp / sum_n] 
            reweight = self.hard_neg_factor * exp_neg / exp_neg.mean()
            sum_exp_neg = (reweight * exp_neg).sum(0, keepdim=True)
            sum_exp_neg *= self.args.num_negatives_by_target[
                self.target_class]
        else:
            sum_exp_neg = exp_neg.sum(0, keepdim=True)
            
        # Compute positive similarities
        anchor_positives = contrastive_batch[:1 + self.n_positives]
        exp_pos = self.compute_exp_sim(model, anchor_positives, 
                                       return_sum=False)
        
        if self.single_pos:
            log_probs = torch.log(exp_pos) - torch.log(sum_exp_neg + exp_pos)
        else:
            log_probs = (torch.log(exp_pos) - 
                         torch.log(sum_exp_neg + exp_pos.sum(0, keepdim=True)))
        loss = -1 * log_probs
        del exp_pos; del exp_neg; del log_probs
        return loss.mean()
    
    def compute_exp_sim(self, model, features, return_sum=True):
        """
        Compute sum(sim(anchor, pos)) or sum(sim(anchor, neg))
        """
        features = features.to(self.args.device)
        if self.arch == 'bert-base-uncased_pt':
            input_ids   = features[:, :, 0]
            input_masks = features[:, :, 1]
            segment_ids = features[:, :, 2]
            outputs = model((input_ids, input_masks, segment_ids, None))
        else:
            outputs = model(features)
        
        sim = self.sim(outputs[0].view(1, -1), outputs[1:])
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        # Should not detach from graph
        features = features.to(torch.device('cpu'))
        outputs = outputs.to(torch.device('cpu'))
        if return_sum:
            sum_exp_sim = exp_sim.sum(0, keepdim=True)
            exp_sim.detach_().cpu(); del exp_sim
            return sum_exp_sim
        return exp_sim
    
    
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
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
    def forward(self, features):
        """
        Compute loss. 
        Args:
        - features (torch.tensor): Input embeddings, expected in form: 
          [target_feature, positive_feature, negative_features[]]
        Returns:
        - loss (torch.tensor): Scalar loss
        """
        target_features = features[0].repeat(features.shape[0] - 2, 1)
        positive_features = features[1].repeat(features.shape[0] - 2, 1)
        loss = self.triplet_loss(target_features, positive_features, features[2:])
        return loss
        
