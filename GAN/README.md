# Generative adversarial network (GAN)
In the project, we consturct a GAN network to generate the images of celebrities. First, we input the images of celebrities and preprocess the images. Second, we construct the GAN network and the hyperparameter have to be set by ourselves. Then, we train the network and show the results of learning curve and the original images and fake images.  
### 1. Data
We construce the dataset of celebrity images by inheriting the Dataset in pytorch. Then, we resize the images into 109*89 (The original images size is 218*179) and transfer the PILImage to tensor.  

### 2. Results
We show the learning curve of the loss including discriminator and generator. Then, we compare the true images and the generating images.  

* Leanrning curve  

<div align="center">
<img src="" height="200px">
<img src="" height="200px">
</div>  

&emsp;
* Sample images  

<div align="center">
<img src="" height="200px">
<img src="" height="200px">
</div>



