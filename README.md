# Vision Transformers - Week 8 Group 2

This weeks papers are about Visual Transformers, particularly:
* **Swin Transformer** - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, Liu, Lin, Cao, Hu, Wei, Zhang, Lin, Guo; 2021](https://arxiv.org/abs/2103.14030)
* **Transformer in Transformer** - [Transformer in Transformer, Han, Xiao, Wu, Guo, Xu, Wang; 2021](https://arxiv.org/abs/2103.00112)
* **Perceiver** - [Perceiver: General Perception with Iterative Attention, Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira; 2021](https://arxiv.org/abs/2103.03206)
* **Peceiver IO** - [	Perceiver IO: A General Architecture for Structured Inputs & Outputs, Jaegle, Borgeaud, Alayrac, Doersch, Ionescu, Ding, Koppula, Zoran, Brock, Shelhamer, Hénaff, Botvinick, Zisserman, Vinyals, Carreira; 2021](https://arxiv.org/abs/2107.14795)
* **MLP-Mixer** - [	MLP-Mixer: An all-MLP Architecture for Vision, Tolstikhin, Houlsby, Kolesnikov, Beyer, Zhai, Unterthiner, Yung, Steiner, Keysers, Uszkoreit, Lucic, Dosovitskiy; 2021](https://arxiv.org/abs/2105.01601)

Datasets used:
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [STL-10](https://cs.stanford.edu/~acoates/stl10/)
* [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet)

**TODO: Other architectures**

## Perceiver IO

### **Code Structure**

The code needed to replicate the Perceiver IO can be found in ``./notebooks/PerceiverIO/PerceiverIOTraining.ipynb``. The code for loading the datasets, creating a base model architecture, and the training code are located in this notebook. I would recommend taking this direct notebook and uploading it to Google Colab for use.

### **Commands to Run the Code**

The code for the experiments is in an ``.ipynb notebook`` that was created on run on colab. It includes the necessary install commands to directly run on colab. 

### **Task**

Use Perceiver IO Architecture for Image Classification

### **Model Architecture**

We used the [pytorch implementation by lucidrains](https://github.com/lucidrains/perceiver-pytorch) of the Perceiver IO architecture and determined the parameters of the model for each task. We also used a randomly generated query vector that had a size of ``(batch_size, 1, num_classes)`` that was jointly trained with the Perceiver IO model. Due to the PyTorch implementation, we reshaped images from (batch_size, num_channels, height, width) to (batch_size, 1, num_channels * height * width) before passing into the model. We used a cross entropy loss function and AdamW optimizer to train the model and the query vector. The output of the model is just a tensor of size (batch_size, output_labels). There were data augmentations also applied to the training images to boost performance. 

### **Conducted Experiments**

We tested out the PerceiverIO model trained from scratch on 3 different datasets. We recorded the training and test loss and accuracies for each dataset. We also ran an ablation study on the affect of learning rate and batch size on the model's performance. 


### **Results**

**CIFAR-10**

Pytorch Perceiver IO Model Parameters:

| Model Parameter | Vaue |
| ------------- | ------------- |
| dim  | 32 * 32 * 3  |
| queries_dim  |10  |
| logits_dim  | 10  |
| depth  | 2  |
| num_latents  | 32  |
| latent_dim  | 64  |
| cross_heads  | 1  |
|latent_heads| 8|
| cross_dim_head  | 128  |
| latent_dim_head  | 128  |


<p float="middle">
  <img src="./notebooks/PerceiverIO/Images/CIFAR10Accuracy.png" width="49%"/>
  <img src="./notebooks/PerceiverIO/Images/CIFAR10Loss.png" width="49%" /> 
</p>

The following hyperparameters were fixed for this experiment: batch_size = 128, learning_rate = 3e-4. We trained a Perceiver IO model from scratch on CIFAR-10 images and were able to achieve a test set accuracy of 0.5619. At about 15 epochs, the test accuracy converged to about 55% accuracy. The model definitely overfit as the training set accuracy continued to grow much after the test set converged. 

**STL-10**

Pytorch Perceiver IO Model Parameters:

| Model Parameter | Vaue |
| ------------- | ------------- |
| dim  | 96 * 96 * 3  |
| queries_dim  |10  |
| logits_dim  | 10  |
| depth  | 2  |
| num_latents  | 96  |
| latent_dim  | 192  |
| cross_heads  | 1  |
|latent_heads| 8|
| cross_dim_head  | 64  |
| latent_dim_head  | 64  |

<p float="middle">
  <img src="./notebooks/PerceiverIO/Images/STL10Accuracy.png" width="49%"/>
  <img src="./notebooks/PerceiverIO/Images/STL10Loss.png" width="49%" /> 
</p>

The following hyperparameters were fixed for this experiment: batch_size = 128, learning_rate = 3e-4. We trained another Perceiver IO model from scratch on the STL-10 dataset. The test set accuracy did not outperform the CIFAR-10 model and achieved a best test accuracy of 0.4748. There is also evidence of overfitting here as the model continued to get higher accuracies on the train data while the test set accuracy had converged.

**TinyImageNet**

Pytorch Perceiver IO Model Parameters:

| Model Parameter | Vaue |
| ------------- | ------------- |
| dim  | 64 * 64 * 3  |
| queries_dim  |200  |
| logits_dim  | 200  |
| depth  | 2  |
| num_latents  | 32  |
| latent_dim  | 64  |
| cross_heads  | 1  |
|latent_heads| 8|
| cross_dim_head  | 64  |
| latent_dim_head  | 64  |

<p float="middle">
  <img src="./notebooks/PerceiverIO/Images/TinyImageNetAccuracy.png" width="49%"/>
  <img src="./notebooks/PerceiverIO/Images/TinyImageNetLoss.png" width="49%" /> 
</p>

The following hyperparameters were fixed for this experiment: batch_size = 128, learning_rate = 3e-4. Our from-scratch Perceiver IO model did poorly on this particularly dataset compared to the two previous models. There was massive overfitting when training this model on the TinyImageNet dataset as the discrepancy between training set and test set accuracy is apparent at the 17th epoch. 

**Batch Size Ablation Study**

We constructed multiple models with CIFAR-10 dataset. We also fixed the learning rate to be 3e-4. The batch sizes tested were 16, 32, 64, 128. The model parameters were fixed for this ablation experiment:

| Model Parameter | Vaue |
| ------------- | ------------- |
| dim  | 32 * 32 * 3  |
| queries_dim  |10  |
| logits_dim  | 10  |
| depth  | 2  |
| num_latents  | 32  |
| latent_dim  | 64  |
| cross_heads  | 1  |
|latent_heads| 8|
| cross_dim_head  | 128  |
| latent_dim_head  | 128  |

<p float="middle">
  <img src="./notebooks/PerceiverIO/Images/BatchSizeAblation.png", width = "50%"/>
</p>

From the above graph, the batch size of Perceiver IO seems to not affect model performance on the CIFAR-10 dataset. The final test accuracies for all of these models was about the same. However, larger batch sizes did lead to slightly better test accuracy. Overall, we would say that the Perceiver IO model is robust to changes in batch size hyperparameter. We tested different batch sizes on the other datasets too which led to similar results as the the above ablation results.

**Learning Rate Ablation Study**

For 12 epochs, we fixed the batch size to 128. The learning rate we tested were 1e-1, 1e-2, 1e-3, and 1e-4. The model parameters are as follows:

| Model Parameter | Vaue |
| ------------- | ------------- |
| dim  | 32 * 32 * 3  |
| queries_dim  |10  |
| logits_dim  | 10  |
| depth  | 2  |
| num_latents  | 32  |
| latent_dim  | 64  |
| cross_heads  | 1  |
|latent_heads| 8|
| cross_dim_head  | 128  |
| latent_dim_head  | 128  |

<p float="middle">
  <img src="./notebooks/PerceiverIO/Images/LRAblation.png", width = "50%"/>
</p>

From the above graph, it appears to be the case that larger learning rates greatly lower performance for the CIFAR-10 dataset. The best test accuracy was found with the lower accuracy of 1e-4 while the worst test accuracy was 0.1108 when learning rate was equal to 1. Therefore, we conclude that the Perceiver IO model is not robust to changes in learning rate and favors larger learning rates during training. 

## **Conclusion**

| CIFAR-10 | STL-10 |  TinyImageNet |
| ------------- | ------------- | -----------|
| 56.19%  | 47.48%  | 12.01% |

Despite data augmentations, there was a good amount of overfitting with a lot of the models, and we did not hit the benchmarks with relatively low overall test accuracies. From the hyperparameter ablation studies, the Perceiver IO model appears to be more robust to batch size and not robust to the learning rate.

Although the results were alright, they could have been better and we believe these are some of the reasons why the Perceiver IO model's performance was subpar:

* **Model Depth**: Lack of computational resources restricted us from training deep versions of this model. We only kept the depth at 2 and believe we can achieve better results if we could train and evaluate with deeper models.
* **Queries**: We used a simpler approach for determining the query vector (specifically a randomized torch tensor). By using more complex methods to determine these queries, we believe we achieve better results.
* **Model Parameters**: Computational resource restrictions also limited our selection of the model parameters. By increasing the mode parameters such as nunmber of dimensions per cross attention and latent self attention heads, it is possible that better results are observed. 
* **Pre-Training**: We could have likely attained better results if we used a pre-trained Perceiver IO model on these datasets compared to the models trained from scratch.

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