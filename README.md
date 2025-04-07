# Diffusion Meets Few-shot Class Incremental Learning

> **[Junsu Kim](https://sites.google.com/view/jjunsss)<sup>1,2\*</sup>, [Yunhoe Ku](https://github.com/samsara-ku)<sup>3</sup>, [Dongyoon Han](https://sites.google.com/site/dyhan0920/)<sup>2†</sup>, [Seungryul Baek](https://sites.google.com/site/bsrvision00/)<sup>1†</sup>** \
> <sub>*Work done during an internship at NAVER AI Lab; † corresponding authors</sub>  
> <sub><sup>1</sup>UNIST, <sup>2</sup>NAVER AI Lab, <sup>3</sup>DeepBrain AI</sub>

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/pdf/2503.23402v1)


## TL;DR 

Our research introduces a novel method for Few-Shot Class-Incremental Learning (FSCIL) that repurposes the text-to-image diffusion model—**Stable Diffusion**—as a frozen generative backbone. We extract multi-scale features via both inversion (forward diffusion) and generation (reverse diffusion) processes, apply class-specific prompt tuning, and incorporate noise-augmented feature replay. This approach not only achieves state-of-the-art performance on benchmarks such as CUB-200, miniImageNet, and CIFAR-100, but it also mitigates catastrophic forgetting while maintaining computational efficiency.
Unlike conventional methods that rely on large-scale supervised pre-training and modifications to backbone architectures, our design preserves the original structure of the diffusion model, providing robust feature representations even for sparsely sampled new classes.


## Abstract
> Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data; while aiming to reduce catastrophic forgetting and learn new information. We propose Diffusion-FSCIL, a novel approach that employs a text-to-image diffusion model as a frozen backbone. Our conjecture is that FSCIL can be tackled using a large generative model’s capabilities benefiting from 1) generation ability via large-scale pre-training; 2) multi-scale representation; 3) representational flexibility through the text encoder. To maximize the representation capability, we propose to extract multiple complementary diffusion features to play roles as latent replay with slight support from feature distillation for preventing generative biases. Our framework realizes efficiency through 1) using a frozen backbone; 2) minimal trainable components; 3) batch processing of multiple feature extractions. Extensive experiments on CUB-200, miniImageNet, and CIFAR-100 show that Diffusion-FSCIL surpasses state-of-the-art methods, preserving performance on previously learned classes and adapting effectively to new ones. 

## Main figure:
<img width="1333" alt="image" src="https://github.com/user-attachments/assets/340f768b-9c92-47b6-8303-4566806bc52b" />

## Core Contributions

1. **Generative Backbone for FSCIL**  
   We introduce a novel framework that uses **Stable Diffusion** as a frozen feature extractor—moving away from traditional discriminative approaches.
   
2. **Multi-Scale Feature Extraction**  
   We systematically extract features at multiple decoder layers in both **inversion** (forward diffusion) and **generation** (reverse diffusion) steps, yielding **rich, complementary** representations.

3. **Class-Specific Prompt Tuning**  
   Instead of generic text prompts, we **optimize prompts** per class to capture fine-grained characteristics. This improves both replay effectiveness and representation quality.

4. **Noise-Augmented Feature Replay**  
   We inject partial noise in the latent space to create **augmented features**, striking a balance between **fidelity** (to original data) and **diversity** (helping generalization).

5. **SOTA Performance & Efficiency**  
   Our framework **outperforms** existing methods on CUB-200, miniImageNet, and CIFAR-100 FSCIL tasks. Despite using a generative model, it remains **computationally efficient** by freezing most diffusion parameters.


## Motivation: Bridging Generative Strengths and FSCIL
- FSCIL demands robust **feature expressiveness** to handle few-shot data while preventing forgetting.  
- **Discriminative backbones** (e.g., ResNet, ViTs, CLIP) often lack the multi-scale, dense representations that come “for free” from diffusion models.  
- Naively using Stable Diffusion (SD) does not lead to effective FSCIL, we began with this observation (see the following figure):

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="480" alt="image"  src="https://github.com/user-attachments/assets/f80ffc10-d687-4ee2-aea4-569a57ba2b5a">
  - SD itself works better than ImageNet-pre-trained ResNet.
  - SD with popular bells and whistles such as 1) generative replays; 2) LoRA does not improve the FSCIL performance

- By **smartly** leveraging SD’s semantic understanding, our method could retain old knowledge while flexibly adapting to new classes—achieving **less forgetting** and **better generalization**.
## Updates

- **2025/03/31**: 🎉 Preprint has been uploaded on ArXiv 🎉.

*(Stay tuned for open-source code and trained weights!)*

## How to Cite

```bibtex
@article{kim2025diffusionfscil,
  title   = {Diffusion Meets Few-shot Class Incremental Learning},
  author  = {Kim, Junsu and Ku, Yunhoe and Han, Dongyoon and Baek, Seungryul},
  journal = {arXiv preprint arXiv:2503.23402},
  year    = {2025}
}
