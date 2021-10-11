# Vision Transformers - Week 8 Group 2

Architectures used:
* **Swin Transformer** - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, Liu, Lin, Cao, Hu, Wei, Zhang, Lin, Guo; 2021](https://arxiv.org/abs/2103.14030)
* **Transformer in Transformer** - [Transformer in Transformer, Han, Xiao, Wu, Guo, Xu, Wang; 2021](https://arxiv.org/abs/2103.00112)
* **Perceiver** - [Perceiver: General Perception with Iterative Attention, Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira; 2021](https://arxiv.org/abs/2103.03206)
* **Peceiver IO** - [	Perceiver IO: A General Architecture for Structured Inputs & Outputs, Jaegle, Borgeaud, Alayrac, Doersch, Ionescu, Ding, Koppula, Zoran, Brock, Shelhamer, Hénaff, Botvinick, Zisserman, Vinyals, Carreira; 2021](https://arxiv.org/abs/2107.14795)
* **MLP-Mixer** - [	MLP-Mixer: An all-MLP Architecture for Vision, Tolstikhin, Houlsby, Kolesnikov, Beyer, Zhai, Unterthiner, Yung, Steiner, Keysers, Uszkoreit, Lucic, Dosovitskiy; 2021](https://arxiv.org/abs/2105.01601)

Datasets used:
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [STL-10](https://cs.stanford.edu/~acoates/stl10/)
* [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet)

# TODO: Other architectures

## MLP-Mixer

### Code
Everything needed to replicate our MLP-Mixer experiments can be found in the [MLP-Mixer notebook](https://github.com/395t/coding-assignment-week-8-vit-2/blob/main/notebooks/MLP_Mixer.ipynb). The [timm PyTorch Image Models collection](https://github.com/rwightman/pytorch-image-models) is used to load the s16-224 MLP-Mixer architecture.

### Experiments: Training from scratch
The model was trained on Colab using Adam with varying batch sizes and learning rates. To save computation time and since our datasets are all fairly small, the small version of the model with 8 layers and patches of size 16x16 was chosen. The model was trained for 20 epochs each time for all datasets and hyperparameters. The datasets are augmented using RandomHorizontalFlip and ColorJitter.

### Results: Training from scratch
#### Tiny Imagenet
The model did not do well at all on Tiny Imagenet, reaching a final validation accuracy of 18% while overfitting on the training data completely. Based on the experiments with different leaning rates and batch sizes, 512 and 0.01 were chosen respectively.

<p float="middle">
  <img src="images/MLP-Mixer_TIN_acc.png" width="49%" />
  <img src="/images/MLP-Mixer_TIN_loss.png" width="49%" /> 
</p>

### STL-10
There is a lot of overfitting on STL-10 as well. The model reaches a final validation accuracy of 44% with a batch size of 256 and a learning rate of 0.0001.

<p float="middle">
  <img src="images/MLP-Mixer_STL_acc.png" width="49%" />
  <img src="/images/MLP-Mixer_STL_loss.png" width="49%" /> 
</p>

### CIFAR-10
Overfitting again on CIFAR. The chosen learning rate is 0.001 and the batch size 512, resulting in a validation accuracy of 59%.

<p float="middle">
  <img src="images/MLP-Mixer_CIFAR_acc.png" width="49%" />
  <img src="/images/MLP-Mixer_CIFAR_loss.png" width="49%" /> 
</p>

## Conclusion: Training from scratch
Even though the data was augmented and the small model used, MLP-Mixer overfit on each dataset and did not yield satisfactory results. The authors promise good results when using models pretrained on ImageNet, which will be examined in the following section.

### Experiments: Fine-Tuning a pretrained model
In this part, a pretrained MLP-Mixer model was fine-tuned on the three datasets. Since no pretrained model was available for the S/16 version, a pretrained B/16 was used instead. This model has 12 instead of 8 layers and more than three times the parameters so a comparison between the two is not really fair. It should prove however, that very good results can be achieved on the small datasets without the massive overfitting seen before.
The model was fine-tuned for two epochs with a learning rate of 0.00005 and a batch size of 32.

### Results: Fine-Tuning a pretrained model
| Dataset | CIFAR-10 | STL-10 | Tiny ImageNet |
| -- | -- | -- | -- |
| Validation Accuracy | 95% | 94.9% | TODO


# Perceiver

## Code
We adopted the publicly available implementation of Peceiver by **lucidrains** from its [Github repository](https://github.com/lucidrains/perceiver-pytorch/).
All of our experiment code can be found in the [**`perceiver`** directory](https://github.com/395t/coding-assignment-week-8-vit-2/blob/main/src/perceiver/). Below are commands to replicate our experiments.

```shell
# Train Perceiver models.
python main.py perceiver_cifar10{,_large,_xl}
python main.py perceiver_stl10{,_large,_xl}
python main.py perceiver_tinyimagenet_large{,_noaug}

# Plot graphs.
```shell
python plot.py plot_train \
    --dataset CIFAR-10 \
    --files result_perceiver_cifar10.yml,result_perceiver_cifar10_large.yml,result_perceiver_cifar10_xl.yml \
    --labels "Perceiver - Medium,Perceiver - Large,Perceiver - XLarge" \
    --out plot_cifar10_train

python plot.py plot_test \
    --dataset CIFAR-10 \
    --files result_perceiver_cifar10.yml,result_perceiver_cifar10_large.yml,result_perceiver_cifar10_xl.yml \
    --labels "Perceiver - Medium,Perceiver - Large,Perceiver - XLarge" \
    --out plot_cifar10_test
```

Replace `CIFAR-10` and `cifar10` with `STL-10` and `stl10` to plot graphs for STL-10.

## Experiment Setup
TODO


## Results
TODO


## Conclusion
TODO