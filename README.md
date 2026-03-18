# Graph Contrastive Learning With Diffusion-Based Transfer (GCLD-CDR)

Official implementation of the paper "Graph Contrastive Learning With Diffusion-Based Transfer for Cross-Domain Recommender System", published in *IEEE Transactions on Systems, Man, and Cybernetics: Systems*.

### 🚀 Overview
GCLD-CDR is a novel cross-domain recommendation framework designed to address two critical bottlenecks in CDR: intra-domain noise and negative transfer.

By combining Graph Contrastive Learning (GCL) with a Diffusion-based transfer mechanism, the model ensures that knowledge shared between domains is both robust and relevant.

**Key Features**
**Denoising Generator**: Prunes unreliable graph edges to refine structural signals.
**Feature Perturbation**: Improves representation diversity via controlled noise injection.
**Diffusion-Based Transfer**: Employs a Gaussian diffusion process and a neural decoder to selectively recover task-relevant information from the source domain.