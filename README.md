# Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations

![Correct-N-Contrast](./assets/cnc_pull_figure.png?raw=true "Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations")

This repository contains code for Correct-N-Contrast (CNC), presented at ICML 2022.  
- Paper link: [https://arxiv.org/abs/2203.01517](https://arxiv.org/abs/2203.01517)  

## Requirements

To install requirements, we recommend setting up a virtual environment with conda:

```setup
conda env create -f environment.yml  
conda activate cnc
``` 

We also include a `requirements.txt` file for installing dependencies with `pip install -r requirements.txt`.  

List of (installable) dependencies:  
* python 3.7.9  
* matplotlib 3.3.2
* numpy 1.19.2  
* pandas 1.1.3  
* pillow 8.0.1  
* pytorch=1.7.0  
* scikit-learn 0.23.2  
* scipy 1.5.2  
* transformers 4.4.2 
* torchvision 0.8.1  
* tqdm 4.54.0  
* umap-learn 0.4.6

## Datasets and code 

**Colored MNIST**: Running the training command below should automatically download and setup the Colored MNIST dataset.  

**Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). Unzipping this should result in a folder `waterbird_complete95_forest2water2`, which should be moved to `./datasets/data/Waterbirds/`.  

**CelebA**: Download dataset files from this [Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). Then move files to `./datasets/data/CelebA/` such that we have the following structure:
```
# In `./datasets/data/CelebA/`:
|-- list_attr_celeba.csv
|-- list_eval_partition.csv
|-- img_align_celeba/
    |-- image1.png
    |-- ...
    |-- imageN.png
```  

**CivilComments-WILDS**: Loading this dataset requires the `transformers` package. One can download the source csv from [here](https://worksheets.codalab.org/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e). Then, move `all_data_with_identities.csv` to `./datasets/data/CivilComments/all_data_with_identities.csv`.


## Training and Evaluation  

For all datasets except Colored MNIST, running the below commands requires loading an initial trained ERM model, which we provide through download links below in the **Pre-trained Models** section. The training for these models is as described in the paper. We recommend downloading as training the initial ERM model can take a fair amount of time, e.g. ~1.5 hours for Waterbirds on a machine with 8 CPUs and 1 NVIDIA V100 GPU, and ~3 hours for CelebA on a machine with 32 CPUs and 4 NVIDIA V100 GPUs. Training these ERM models also just involves standard mini-batch SGD with the dataset labels.


### Colored MNIST  

Train:  
```train
python train_supervised_contrast.py --dataset colored_mnist --train_encoder --arch cnn --data_cmap hsv --test_shift random -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9 --p_correlation 0.995 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 --max_epoch_s 5 --bs_trn_s 32 --num_anchor 32 --num_positive 32 --num_negative 32 --num_negative_easy 32 --batch_factor 32 --optim sgd --lr 1e-3 --momentum 0.9 --weight_decay 1e-4 --weight_decay_c 1e-4 --target_sample_ratio 1 --temperature 0.05 --max_epoch 3 --no_projection_head --contrastive_weight 0.75 --bs_trn 32 --bs_val 32 --num_workers 0 --no_projection_head --log_loss_interval 10 --checkpoint_interval 10000 --log_visual_interval 40000 --verbose --replicate 42 --seed 42
```

Evaluate:  
```evaluate
python train_supervised_contrast.py --dataset colored_mnist --arch cnn --evaluate --load_encoder [model_file_name.pth.tar] --data_cmap hsv --test_shift random -tc 0 1 -tc 2 3 -tc 4 5 -tc 6 7 -tc 8 9 --p_correlation 0.995 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0 -tcr 1.0
```

### Waterbirds

Train:  
```train
python train_supervised_contrast.py --dataset waterbirds --arch resnet50_pt --train_encoder --pretrained_spurious_path "./model/waterbirds/waterbirds_erm_regularized.pt" --num_anchor 17 --num_positive 17 --num_negative 17 --num_negative_easy 17 --batch_factor 32 --optim sgd --lr 1e-4 --momentum 0.9 --weight_decay 1e-3 --weight_decay_c 1e-3 --target_sample_ratio 1 --temperature 0.1 --max_epoch 5 --no_projection_head --contrastive_weight 0.75 --log_visual_interval 10000 --checkpoint_interval 10000 --verbose --log_loss_interval 10 --replicate 42 --seed 42
```

Evaluate:  
```evaluate
python train_supervised_contrast.py --dataset waterbirds --arch resnet50_pt --evaluate --load_encoder waterbirds_cnc_pretrained.pth.tar 
```

### CelebA

Train:  
```train
python train_supervised_contrast.py --dataset celebA --arch resnet50_pt --train_encoder --pretrained_spurious_path "./model/celebA/celeba_erm_regularized.pt" --num_anchor 64 --num_positive 64 --num_negative 64 --num_negative_easy 64 --batch_factor 32 --optim sgd --lr 1e-5 --momentum 0.9 --weight_decay 1e-1 --weight_decay_c 1e-1 --target_sample_ratio 0.1 --temperature 0.05 --max_epoch 15 --no_projection_head --contrastive_weight 0.75 --log_visual_interval 10000 --checkpoint_interval 10000 --verbose --log_loss_interval 10 --replicate 42 --seed 42
```

Evaluate:  
```evaluate
python train_supervised_contrast.py --dataset celebA --arch resnet50_pt --evaluate --load_encoder celebA_cnc_pretrained.pth.tar 
```

### CivilComments-WILDS

Train:  
```train
python -W ignore train_supervised_contrast.py --dataset civilcomments --arch bert-base-uncased_pt --train_encoder --pretrained_spurious_path ./model/civilcomments/civilcomments_erm_early.pth.tar --num_anchor 16 --num_positive 16 --num_negative 16 --num_negative_easy 16 --batch_factor 128 --bs_trn 16 --clip_grad_norm --optim AdamW --lr 1e-4 --weight_decay 1e-2 --target_sample_ratio 0.1 --temperature 0.1 --max_epoch 10 --no_projection_head --contrastive_weight 0.75 --log_loss_interval 10 --checkpoint_interval 10000 --verbose --log_visual_interval 400000 --verbose --replicate 42 --seed 42
```

Evaluate:  
```eval
python -W ignore train_supervised_contrast.py --dataset civilcomments --arch bert-base-uncased_pt --evaluate --load_encoder civilcomments_cnc_pretrained.pth.tar 
```

## Pre-trained Models

Both pretrained initial ERM models and the trained Correct-N-Contrast models are available to download [here](https://drive.google.com/drive/folders/1SqhrdhLbBCNmTqCEf9Xm5ib4iTnt5ncx?usp=sharing). 

Once downloaded, models should be moved to the following file paths:  

**Waterbirds**  
- ERM model: `./model/waterbirds/./model/waterbirds/waterbirds_erm_regularized.pt`  
- CNC model: `./model/waterbirds/config-tn=waterbird_complete95-cn=['forest2water2']/waterbirds_cnc_pretrained.pth.tar`  

**CelebA**  
- ERM model: `./model/celebA/celeba_erm_regularized.pt`
- CNC model: `./model/celebA/config/celebA_cnc_pretrained.pth.tar`

**CivilComments-WILDs**  
- ERM model: `./model/civilcomments/civilcomments_erm_early.pth.tar`
- CNC model: `./model/civilcomments/config/civilcomments_cnc_pretrained.pth.tar`
