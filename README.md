# DeepLearning Model Implementation

## Envirment & Requirements

* `Rtx 3070`
* `TensorFlow >= 2.4.1`

## Multi-Label Classification

* `colab`
* `TensorFlow >= 2.4.1`
* [Dataset](https://dacon.io/competitions/official/235697/data/)
* [EfficientNet-b3](https://github.com/pwr4779/DeepLearning/blob/master/NoisyEMNISTClassification/EfficientNet-b3.py)
* [Resnet101_32x8d](https://github.com/pwr4779/DeepLearning/blob/master/NoisyEMNISTClassification/Resnet101_32x8d.py)
* colab gpu ram : 16G  

| <center>Model</center> | <center>Acc</center> |  
|:------|---:|  
| EfficentNet-b0 | 0.78586 |  
| EfficentNet-b2 | 0.8198 |  
| EfficentNet-b3 | 0.828 |  
| Resnet101_32x8d | 0.84738 |  

* private score : 0.84738 (46/811)
* best_model : Resnet101_32x8d
* Resnet101 batch_size = 32
* Auggumenation  
```python
a_train = A.Compose([
    A.HorizontalFlip(0.5),
    A.RandomRotate90(0.5),
    A.VerticalFlip(0.5),
    A.OneOf([
       A.GaussNoise(var_limit=[10, 50]),
       A.MotionBlur(),
       A.MedianBlur(),
   ], p=0.2),
   A.OneOf([
       A.OpticalDistortion(distort_limit=1.0),
       A.GridDistortion(num_steps=5, distort_limit=1.),
       A.ElasticTransform(alpha=3),
   ], p=0.2),
    A.OneOf([
      A.JpegCompression(),
  ], p=0.2),
    A.Cutout(num_holes=10, max_h_size=5, max_w_size=5, always_apply=False, p=0.5, ),
    # DiagonalReverse(),
])
```  
* TTA is not used since accuracy is downed after TTA implementation

## Variational AutoEncoder
- [Variational AutoEncoder](https://github.com/pwr4779/DeepLearning/blob/master/VAE/variationalAutoencoder.py)  
tensorflow implementation of Variational AutoEncoder from the [paper](https://arxiv.org/pdf/1606.05908.pdf).

### result
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/VAE/vae.png"/>
    </td>
</tr>
</table>

## Undercomplete
- [Undercomplete](https://github.com/pwr4779/DeepLearning/blob/master/VAE/Undercomplete.ipynb)

## Stacked AutoEncoder
- [Stacked AutoEncoder](https://github.com/pwr4779/DeepLearning/blob/master/AutoEncoder/Stacked%20AutoEncoder.ipynb)

## Emnist Using Denoising AutoEncoder
- [Dataset](https://www.kaggle.com/crawford/emnist)
### result
<table border="0">
<tr>
    <th>original image</th>
    <th>Denoising image(Input)</th>
    <th>Deonising AutoEncoder(Output)</th>
</tr>
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/EMNIST-using-Denoising-AutoEncoder/content/image.jpg" width="100%"/>
    </td>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/EMNIST-using-Denoising-AutoEncoder/content/noisy.jpg" width="100%"/>
    </td>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/EMNIST-using-Denoising-AutoEncoder/content/test.jpg" width="100%"/>
    </td>
</tr>
</table>  

## GAN
- [Original Paper Link](https://arxiv.org/abs/1406.2661) / [code](https://github.com/pwr4779/DeepLearning/blob/master/GAN/GAN/gan.ipynb)

### result 
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/GAN/GAN/epoch1000.JPG" width="100%" />
    </td>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/GAN/GAN/epoch2000.JPG", width="100%" />
    </td>
</tr>
</table>

## DCGAN  
- [Original Paper Link](https://arxiv.org/abs/1511.06434) / [code](https://github.com/pwr4779/DeepLearning/blob/master/GAN/DCGAN/DCGAN.ipynb)

### result
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/GAN/DCGAN/dcgan.gif"/>
    </td>
</tr>
</table>

## Conditional Generative Adversarial Network (CGAN)

### Abstract  
Generative Adversarial Nets were recently introduced as a novel way to train
generative models. In this work we introduce the conditional version of generative
adversarial nets, which can be constructed by simply feeding the data, y, we wish
to condition on to both the generator and discriminator. We show that this model
can generate MNIST digits conditioned on class labels. We also illustrate how
this model could be used to learn a multi-modal model, and provide preliminary
examples of an application to image tagging in which we demonstrate how this
approach can generate descriptive tags which are not part of training labels.

### Architecture_diagram
![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/Architecture_diagram.png)

### Generator Embedding
![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/generator_embedding.jpg)

### Discriminator Ebedding
![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/discriminator_embedding.jpg)

- [Original Paper Link](https://arxiv.org/abs/1411.1784) / [code](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/CGAN.ipynb)

### result
| Epoch       |  predict |
:-------------------------:|:-------------------------:
15999  |  ![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/result/result_15999.png)
16999  |  ![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/result/result_16999.png)
17999  |  ![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/result/result_17999.png)
18999  |  ![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/result/result_18999.png)
19999  |  ![](https://github.com/pwr4779/DeepLearning/blob/master/GAN/CGAN/result/result_19999.png)

## Style Transfer - ResNet50
- [Style Transfer](https://github.com/pwr4779/DeepLearning/blob/master/ResNet-StyleTransfer/ResNet-StyleTransfer.ipynb)

<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/ResNet-StyleTransfer/cat.jpg" width="100%" />
    </td>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/ResNet-StyleTransfer/stylized-image.png", width="100%" />
    </td>
</tr>
</table>

## tf.GradientTape(), Tensorboard manual MNIST
- [GradientTape](https://github.com/pwr4779/DeepLearning/blob/master/Tensorflow%20Advanced%20Tutorials/MNIST.ipynb)
