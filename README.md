# DDIM for inverse problem
Provably solving inverse problems with diffusion prior through DDIM-type sampler

### Abstract

Diffusion-based inverse problem solvers have demonstrated strong empirical performance across a wide range of applications. However, many existing methods either lack rigorous theoretical guarantees or require computationally intensive modifications to the sampling procedure. In this work, we propose a simple and efficient algorithm for solving linear inverse problems with diffusion priors via a DDIM-type sampler. Our method requires only lightweight coordinate-wise modifications to DDIM while explicitly incorporating the measurement structure. The key insight is that, along each singular direction of the measurement operator, posterior sampling can be carried out by a DDIM-type update that follows the learned prior when the observation is unreliable and switches to a calibrated measurement-based predictor once the observation signal-to-noise ratio exceeds the diffusion signal-to-noise ratio. We prove that the proposed sampler converges to the posterior distribution of the signal conditioned on the measurements. Overall, our framework shows that posterior sampling in noisy linear inverse problems can be reduced to simple, coordinate-wise DDIM updates, yielding an algorithm that is easy to implement, computationally efficient, and provably consistent with the Bayesian posterior.

## Requirements

Please refer to [`environment.yaml`](./environment.yaml) for the complete list of dependencies.

You can create and activate the environment using Miniconda:

```bash
conda env create -f environment.yaml -n DDIM_inv
conda activate DDIM_inv
```

## Pre-trained Models
We evaluate our method using publicly available pre-trained diffusion models:
1. ImageNet: https://github.com/openai/guided-diffusion ([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))
2. CelebA: https://github.com/ermongroup/SDEdit (https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link)
Place them into exp/logs/[dataset_name] respectively.

## Getting started

To reproduce the main results from our paper, simply run:

```bash
sh evaluation.sh
```
