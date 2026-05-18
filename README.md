# FeaSplit: Feature-Level Data Partitioning for Federated Learning in Indoor Localization

## 🔍 Overview

Federated Learning (FL) has emerged as a promising paradigm for privacy-preserving distributed modeling. However, its performance is significantly affected by data heterogeneity across clients.

In indoor localization tasks, this issue becomes particularly critical, as accurate positioning is essential for applications such as smart buildings, healthcare monitoring, and emergency response. Existing Non-IID partitioning strategies are primarily label-driven and fail to capture the intrinsic feature diversity of RSS fingerprint data, especially for regression tasks like latitude–longitude prediction.

To address this limitation, we propose **FeaSplit**, a feature-level data partitioning framework that constructs more realistic and controllable heterogeneous client distributions.

---

## 🚀 Key Features

- **Feature-level partitioning** instead of label-based splitting
- **Subspace-aware clustering** to model intrinsic data structure
- Supports **regression tasks** (e.g., indoor localization)
- Generates **realistic and controllable heterogeneity**
- Compatible with standard FL frameworks

---

## 🧠 Method

FeaSplit consists of two main components:

1. **Feature Extraction**  
   Pre-trained models are used to obtain high-level feature representations of raw RSS data.

2. **Subspace-Regularized Clustering**  
   Data is partitioned by jointly considering:
   - Euclidean distance (centroid similarity)
   - Subspace projection error (feature structure consistency)

Each client is assigned data samples that share similar feature subspaces, leading to more realistic heterogeneity.
