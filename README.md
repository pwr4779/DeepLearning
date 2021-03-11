# DeepLearning

## Envirment & Requirements

* `Rtx 3070`
* `TensorFlow >= 2.4.1`

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

## DCGAN
- [DCGAN](https://github.com/pwr4779/DeepLearning/blob/master/GAN/DCGAN/DCGAN.ipynb)

### result
<table border="0">
<tr>
    <td>
    <img src="https://github.com/pwr4779/DeepLearning/blob/master/GAN/DCGAN/dcgan.gif"/>
    </td>
</tr>
</table>

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

## Multi-Label Classification
* `colab`
* `TensorFlow >= 2.4.1`
* [Dataset](https://dacon.io/competitions/official/235697/data/)
* private score : 0.84738 (46/811)
* model : Resnet101_32x8d
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

## tf.GradientTape(), Tensorboard manual MNIST
- [GradientTape](https://github.com/pwr4779/DeepLearning/blob/master/Tensorflow%20Advanced%20Tutorials/MNIST.ipynb)
