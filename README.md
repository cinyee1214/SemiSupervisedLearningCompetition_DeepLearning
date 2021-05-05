# Final Competition of Deep Learning 2021

## Overview
This is a  competition in which we will  compete  with  our  classmates  on  who  can find the best self/semi-supervised learning algorithm. We are given a dataset with a large amount of unlabeled data and a small amount of labeled data to train the model, and the final performance of our model will be evaluated ona hidden test set and posted on a public leaderboard.

## Dataset
The dataset, of color images of size 96×96, that has the following structure:
- 512, 000 unlabeled images,
- 25, 600 labeled training images (32 examples, 800 classes),
- 25, 600 labeled validation images (32 examples, 800 classes).

---

## Sbatch
There are 5 sbatch files included in the **sbatch** folder
- pretrain.sbatch
- frozen.sbacth
- unfrozen.sbatch
- extra_frozen.sbatch
- extra_unfrozen.sbatch

these sbatch files contain all the command we used to create our two final models.

---

## Replication
### Pretrain
We used **unsupervised.py** to pretrain the model. Below are the hyper parameters we used. The values end with a `*` are the default values.


| Name | Value |
|:----:|:-----:|
| Backbone | Resnet152 |
| Epochs | 250 |
| Batch Size | 256* |
| Learning Rate | 0.03* |
| SGD Momentum | 0.9* |
| Weight Decay | 1e-4* |
| Cosine Adjust LR | True |

And below are the MoCo related hyper parameters.

| Name | Value |
|:----:|:-----:|
| Output Dim | 128* |
| Queue Size | 65536* |
| Momentum | 0.999* |
| Temperature | 0.07* |
| MLP | True |

You can see from the **pretrain.sbatch** file the command to run **unsupervised.py**.

```sh
python $HOME/dl05/unsupervised.py -a resnet152 --epochs 250 --resume $SCRATCH/resnet152/$LATEST_CP --checkpoint $SCRATCH/resnet152/ --mlp --cos
```

`--resume` specifies the checkpoint we used to continue training, and `--checkpoint` specifies the folder to save the checkpoints.

### Downstream
We run **frozen.py** for 100 epochs to do the classification with frozen features.

| Name | Value |
|:----:|:-----:|
| Arch | Resnet152 |
| Epochs | 100* |
| Batch Size | 256* |
| Learning Rate | 30* |
| SGD Momentum | 0.9* |
| Weight Decay | 0* |
| Cosine Adjust LR | True |

You can see from the **frozen.sbatch** file the command to run **frozen.py**.

```sh
python $HOME/dl05/frozen.py -a resnet152 --pretrained $SCRATCH/resnet152/checkpoint_0249.pth.tar --checkpoint $SCRATCH/frozen/ --cos

```
`--pretrained` specifies the path of the pretrained parameters.

And then we run **unfrozen.py** for another 100 epochs to further the training with unfrozen features. The hyper parameters here are mostly the same as above except for the learning rate.

| Name | Value |
|:----:|:-----:|
| Learning Rate | 0.003 |

You can see from the **unfrozen.sbatch** file the command to run **unfrozen.py**.

```sh
python $HOME/dl05/unfrozen.py -a resnet152 --lr 0.003 --pretrained $SCRATCH/frozen/best_0099.pth.tar --checkpoint $SCRATCH/unfrozen/ --cos
```

`--pretrained` specifies the path of the model parameters with the best validation accuracy from the frozen features phase.

---

## Extra Labels
### Labeling Request
The code for labeling request is under the **request indicies** folder. You can find two python files: **prob.py** and **request_labels.py**. We run these python files on local machine so there's no sbatch file for them.

**prob.py** generates the probabilities the model gives of 800 classes for every unlabeled images. Then **request_labels.py** will generate labeling request according to the probabilities.
### Use of Extra Labels
For extra labels, we used them with the same training methods as above. Note that we used `CustomDatasetPlus` to load the dataset, which you can see from **dataloaderP.py** under **custom** folder.

First load models from pretrained phase and run **extra_frozen.py** for 100 epochs. The hyper parameters here are the same as frozen features phase without extra labels. From **extra_frozen.sbatch** you can see how to run **extra_frozen.py**.
```sh
python $HOME/dl05/extra_frozen.py -a resnet152 --pretrained $SCRATCH/resnet152/checkpoint_0249.pth.tar --checkpoint $SCRATCH/extraf/ --cos
```

As before, we run another 100 epochs with unfrozen features. The hyper parameters here are the same as the one without extra labels. From **extra_unfrozen.sbatch** you can see how to run **extra_unfrozen.py**.

```sh
python $HOME/dl05/extra_unfrozen.py -a resnet152 --lr 0.003 --pretrained $SCRATCH/extraf/best_0095.pth.tar --checkpoint $SCRATCH/extrauf/ --cos
```

`--pretrained` specifies the path of the model parameters with the best validation accuracy from the frozen features with extra labels phase.


