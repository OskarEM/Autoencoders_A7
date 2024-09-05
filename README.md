# Deep Autoencoders and Generative Networks

This project implements **Deep Autoencoders** and **Generative Adversarial Networks (GANs)** to handle image processing tasks such as image reconstruction and generation. The project explores two main tasks: using autoencoders for image reconstruction and GANs for generating new images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Deep Autoencoders](#deep-autoencoders)
  - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build **deep autoencoders** for image reconstruction and **Generative Adversarial Networks (GANs)** for image generation. We used TensorFlow/Keras libraries for both tasks, focusing on training the models on a dataset of images with added stamps and comparing the reconstructed or generated images.

## Dataset

The dataset consists of images that have been processed to include a stamp ("September 11, 2023") on each image. The images are split into **training** and **validation** sets, with stamped versions and their original counterparts used for training the models.

## Methodology

### Deep Autoencoders

1. **Preprocessing**:
   - The dataset was preprocessed by adding a stamp to each image, resizing them, and normalizing pixel values to a range of **0-1**.
   - Two datasets were created: one with original images and one with stamped images.

2. **Model Architecture**:
   - The autoencoder was built using convolutional layers for the encoder and decoder.
   - The encoder compresses the input image to a latent space representation, while the decoder reconstructs the image from this compressed representation.
   - The model was trained using **Adam optimizer** and **mean squared error (MSE)** as the loss function.

3. **Training**:
   - The autoencoder was trained on the image dataset, with early stopping to prevent overfitting.
   - Two versions of the autoencoder were trained: one with higher quality but longer training time, and another with faster training but slightly lower image quality.

### Generative Adversarial Networks (GANs)

1. **Preprocessing**:
   - Similar preprocessing steps were used for the GAN model, including loading, resizing, and normalizing images.
   
2. **Model Architecture**:
   - **Generator**: A model that generates new images from random noise. The architecture includes dense layers, **Conv2DTranspose layers** for upscaling, **Batch Normalization**, and **LeakyReLU** activation.
   - **Discriminator**: A model that classifies images as real or fake. It uses **Conv2D layers**, **LeakyReLU**, and **Dropout** for downscaling.
   - **GAN**: The generator and discriminator models are trained together, where the discriminator learns to classify real and fake images, and the generator learns to fool the discriminator.

3. **Training**:
   - The GAN was trained using alternating steps: training the discriminator on real and generated images and training the generator to improve its ability to generate convincing images.
   - Techniques such as **label smoothing** and adding noise were used to improve training stability.

## Results

### Autoencoder:
- Two models were trained:
  1. **Model 1**: Longer training, resulting in higher image reconstruction quality.
  2. **Model 2**: Shorter training, but with more blurred image reconstructions.


## Conclusion

This project demonstrated the use of **autoencoders** for image reconstruction and **GANs** for image generation. The autoencoder was effective in reconstructing the original images from stamped versions. Improvements could be made by further fine-tuning the models and increasing the dataset size.


