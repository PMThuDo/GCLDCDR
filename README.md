# Graph Contrastive Learning With Diffusion-Based Transfer (GCLD-CDR)

Official implementation of the paper "Graph Contrastive Learning With Diffusion-Based Transfer for Cross-Domain Recommender System", published in *IEEE Transactions on Systems, Man, and Cybernetics: Systems*.

### 📖 Introduction
Cross-Domain Recommender Systems (CDRSs) aim to alleviate data sparsity by transferring knowledge from a source domain to a target domain. However, they often struggle with **intra-domain noise** and **negative transfer**.

**GCLD-CDR** addresses these by:
- **1. Intra-domain Refinement**: Using Graph Contrastive Learning (GCL) with structural denoising and feature perturbation.
- **2. Inter-domain Transfer**: Utilizing a Diffusion-based mechanism to filter noise during the knowledge transfer process.

### 🛠️ Key Components

**1. Graph Contrastive Learning (GCL) Module**

To improve the quality of user/item representations within a single domain, we employ two augmentation strategies:

- *Feature Perturbation Generator*: Introduces controlled noise to create diverse representation views.

- *Denoising Generator*: Learns to prune unreliable or "noisy" edges in the interaction graph to focus on high-confidence signals.

**2. Diffusion-Based Transfer Mechanism**

To prevent negative transfer (where irrelevant source data hurts target performance), we treat knowledge transfer as a generative denoising task:

- *Forward Process*: Source-domain user representations are gradually perturbed using a Gaussian diffusion process.

- *Reverse Process (Neural Decoder)*: The model learns to reverse the diffusion, selectively recovering only the features relevant to the target task while filtering out misaligned signals.

### 🛠️ Architecture
The framework consists of two primary stages:

**1. Intra-Domain Enhancement**: Two complementary augmentation modules (structural denoising and feature perturbation) optimize representations within a single domain.

**2. Inter-Domain Transfer**: A diffusion-based mechanism that progressively perturbs source representations and reverses the process to filter out misaligned signals, preventing negative transfer.

### 📝 Citation

If you find this work useful in your research, please consider citing our paper:

@article{do2025graph,
  title={Graph Contrastive Learning With Diffusion-Based Transfer for Cross-Domain Recommender System},
  author={Do, Pham Minh Thu and Zhang, Qian and Zhang, Guangquan and Lu, Jie},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={56},
  number={1},
  pages={375--386},
  year={2025},
  publisher={IEEE}
}