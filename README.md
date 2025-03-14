# Image-Enhancement-Project
This Git Repo is used for managing our Image Enhancement Project for a group assignment at FPT University.

# Training Process
For further detail, checkout [train_esrgan.ipynb](train_esrgan.ipynb). This process is based on the papers [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) and [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833).
## Constants
- EPOCHS = 10
- LR_SIZE = (96, 96)
- HR_SIZE = (384, 384)
- BATCH_SIZE = 4

## Model(s)

We are choosing [ESRGAN](https://arxiv.org/abs/1809.00219) as our model for Image Enhancement Project.
### Generator
- Overall Architecture

![overall.png](images/model/generator/overall.png)
<a href="https://arxiv.org/pdf/1809.00219#page=5"><p style='text-align: center'>Figure 3 from ESRGAN paper</p></a>

- Basic Block: RRDB (Residual in Residual Dense Block)

![RRDB.png](images/model/generator/RRDB.png)
<a href="https://arxiv.org/pdf/1809.00219#page=5"><p style='text-align: center'>Figure 4 from ESRGAN paper</p></a>

### Discriminator
We are using UNet as our discriminator. This is used in the paper [Real-ESRGAN](https://arxiv.org/abs/2107.10833).
- Overall Architecture

![unet.png](images/model/discriminator/unet.png)
<a href="https://arxiv.org/pdf/2107.10833#page=5"><p style='text-align: center'>Figure 6 from Real-ESRGAN paper</p></a>

## Loss Functions
### Discriminator Loss:
![discriminator_loss.png](images/loss/discriminator_loss.png)
Where:
+ $`\mathbb{E}_{x_r}`$ is the
operation of taking average for all real data in the mini-batch. $`\mathbb{E}_{x_f}`$ is the operation of taking average for all fake data in the mini-batch.
+ $`D_{Ra}(x_r, x_f)=σ(C(x_r)−E_{x_f}[C(x_f)])`$ with $`σ(x)`$ is the Sigmoid Function and $`C(x)`$ is the non-transformed discriminator output.
### Generator Loss:
![generator_loss.png](images/loss/generator_loss.png)
Where:
+ $`L_percep`$ is the Perceptual Loss calculated using pre-trained VGG19-54, where 54 indicates features obtained by the 4th convolution before the 5th maxpooling layer.
+ $`L_G^{Ra}`$ is the Generator Adversarial Loss
![adversarial_loss.png](images/loss/adversarial_loss.png)
+ $L_1$ is the L1 Loss.
+ Parameter: $\lambda = 5 \times 10^{-3}$ and $\eta = 1 \times 10^{-2}$.
## Optimizers
We used Adam as optimizers for our models.
- Generator: learning_rate=2e-4, beta_1=0.9, beta_2=0.99
- Discriminator: learning_rate=1e-4

## Metrics
We used PSNR (Peak Signal to Noise Ratio) and SSIM (Structural Similarity Index Measure)
- PSNR:

![psnr.png](images/metrics/psnr.png)

- SSIM:

![ssim.png](images/metrics/ssim.png)

- MSE:

![mse.png](images/metrics/mse.png)

# Result
For further detail, checkout [train_esrgan.ipynb](train_esrgan.ipynb) and [test_model.ipynb](test_model.ipynb).

## Train Result
### Generator Loss
![g_loss.png](images/result/g_loss.png)
### Discriminator Loss
![d_loss.png](images/result/d_loss.png)
### PSNR
![psnr.png](images/result/psnr.png)
### SSIM 
![ssim.png](images/result/ssim.png)

## Demo Result
![image_comparison.png](images/result/image_comparison.png)
- PSNR = 21.770206451416016
- SSIM = 0.6498551964759827










