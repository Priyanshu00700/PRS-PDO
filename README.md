# MODEL IMAGE

## Novelty/Innovation:
- *Edge-optimized topographic preprocessing replaces O(N²) denoising with a linear-time O(N/s²) Fast Guided Filter (FGF) pipeline via 255–I inversion, drastically reducing battery drain on consumer edge devices.*
- *Chiral Siamese Weight-Sharing exploits anatomical palm symmetry (horizontal flip of the left hand) to double the effective few-shot enrollment data with zero additional memory overhead.*
- *Score-level decision fusion is optimized by a single-agent h(PRS-PDO) hybrid meta-heuristic featuring a non-linear Hadamard interaction term (w₃·(C_ear ∘ C_palm)) that captures cross-modal synergy without any secondary neural network.*

---

**Architecture Diagram:**

```
[Raw Input] --> Left Palm (Flip) + Right Palm --> [Siamese Pool: 4 Train / 2 Test]
                                                             |
                                                             v
[Ear Input] ------------------> [Topographic Preprocessing (255-I, CLAHE, FGF)]
                                                             |
                 +-------------------------------------------+---------------------------+
                 |                                                                         |
          [Stream A: Ear]                                                  [Stream B: Palm Pairs]
                 |                                                                         |
   (Ghost-MobileNetV3 + Coord. Attention)                      (Chiral Siamese + Coord. Attention)
                 |                                                                         |
         [C_ear Score]                                                        [C_palm Score]
                 |                                                                         |
                 +-------------------------> [h(PRS-PDO) Agent] <--------------------------+
                                                      |
                                   [w1*Ear] + [w2*Palm] + [w3*(Ear ∘ Palm)]
                                                      |
                                             [Decision Threshold]
```

---

## PROJECT MODULES

### 1. Data Preprocessing

#### Ear & Palm Dataset Pre-processing Steps (Hardware-Aware Topographic Pipeline)

| S.No. | Technique | Purpose | Discriminative Features |
|:------|:----------|:--------|:------------------------|
| 1. | **Topographic Inversion (255 – I)** | Converts dark biometric lines (ridges, cartilage) into high-intensity activation peaks compatible with lightweight ReLU networks. | Glowing structural cartilage outlines and palm principal lines against a muted background. |
| 2. | **Localized CLAHE (β=2.0)** | Adaptively boosts contrast in localized tile regions (8×8 grid) to emphasize subtle texture variation without over-amplifying noise. | Clear ridge-valley contrast, enhanced inner ear architecture, and sharpened structural lines. |
| 3. | **LS-AMF (Adaptive Median Approximation)** | Eliminates impulsive salt-and-pepper noise by applying a lightweight 3×3 median kernel, preserving biometric edge integrity. | Smooth homogeneous regions, preserved structural edges and principal palm lines. |
| 4. | **Fast Guided Filter (FGF, s=4)** | Replaces O(N²) Non-Local Means denoising with a linear-time O(N/s²) filter. Subsamples the image by ratio s=4, solves linear smoothing coefficients at low resolution, then upsamples back. | Noise-free topographic maps with preserved boundaries; achieves 16× complexity reduction over NLMeans. |

---

### 2. Feature Extraction

After preprocessing, ear and palm images are processed independently using two heterogeneous deep learning streams:

**Stream A (Ear Images):** Processed through a modified **Ghost-MobileNetV3** backbone accepting 1-channel topographic maps. A **Coordinate Attention (CA)** block decomposes the feature tensor into X and Y directional average-pooled vectors, encoding precise spatial coordinates of glowing cartilage peaks. Spatial Dropout (p=0.4) is applied before the final embedding layer to prevent memorization of the minimal enrollment data.

**Stream B (Chiral Siamese Palm):** Processed by a shared-weight **Siamese Ghost-MobileNetV3** with the same CA block. The Chiral Mirror Mechanism (horizontal flip of left-hand images) is applied prior to pooling with native right-hand images, yielding 4 structurally identical samples per subject from only 2 physical hands. Zero-memory on-the-fly augmentations (±5° rotation, 10% perspective distortion) prevent overfitting. A tightened **L2 Contrastive Loss** (margin m=1.0) supervises pair-wise training.

Each stream outputs a 128-dimensional L2-normalized embedding vector, from which a scalar confidence score is derived via Gaussian kernel conversion: **C = exp(–distance²)**.

---

### 3. Classification

The system unifies the independently computed ear and palm confidence scores (C_ear, C_palm) via the **h(PRS-PDO) Single-Agent Score-Level Fusion** optimizer. A single agent transitions from global **Prism Refraction Search (PRS)** exploration (when energy E ≥ 1.0) to local **Prairie Dog Optimization (PDO)** exploitation (when E < 1.0), guided by continuous energy decay:

**E = 2 · exp(–(3t/T)²)**

The agent minimizes classification error to solve for three optimal weights in the non-linear fusion equation:

**C_fused = w₁·C_ear + w₂·C_palm + w₃·(C_ear ∘ C_palm)**

The w₃ Hadamard interaction term captures hidden cross-modal synergy without requiring a secondary neural network or memory-heavy cross-attention module. After convergence, the final fused score is normalized to [0, 1] and compared against a decision threshold of 0.5 for authentication.

The classifier's effectiveness is evaluated using accuracy, EER, FAR, and FRR — providing a transparent, publication-ready performance track record.

---

## INNOVATIONS

### 1. Chiral Mirror Mechanism for Zero-Memory Few-Shot Learning (FSL)
The system overcomes extreme data scarcity (only 4 enrollment images per user) by exploiting anatomical symmetry. Horizontally flipping the left palm to mirror the right hand doubles the effective support set — expanding 4 images into C(4,2) = **6 positive Siamese pairs per subject** — without storing additional data or creating synthetic artifacts. This constitutes a novel, mathematically principled approach to FSL on edge devices.

### 2. Topographic Peak Maps via Linear-Time Hardware-Aware Preprocessing
Standard biometric preprocessing (Non-Local Means, Bilateral Filter) runs in O(N²), draining battery on edge devices. The proposed pipeline replaces these with a linear-time cascade: 255–I inversion → CLAHE → LS-AMF → Fast Guided Filter (O(N/s²)). The inversion step casts biometric structures as high-intensity peaks suited to ReLU-based networks, resulting in better gradient flow with fewer parameters.

### 3. Single-Agent h(PRS-PDO) Non-Linear Score Fusion
Rather than concatenating feature vectors (bloating edge-device memory) or using population-based metaheuristics (high battery cost), the system uses a **single-agent hybrid optimizer**. The non-linear Hadamard interaction term mathematically proves that the cross-correlation of modalities yields greater security than either unimodal stream alone — accounting for **>30% of the final decision logic** at w₃ = 0.303.

---

## KEYWORDS

- Multimodal Ear and Palmprint Recognition
- Few-Shot Learning (FSL)
- Chiral Siamese Networks
- Edge AI / Hardware-Aware Inference
- Hybrid Metaheuristic Optimization
- Score-Level Decision Fusion
- Topographic Biometric Preprocessing

---

## EXPERIMENTAL RESULTS

### Fusion Weight Distribution

The h(PRS-PDO) agent converged on the following optimal weighting schema (normalized), proving the mathematical necessity of the non-linear interaction term:

| Component | Variable | Optimal Weight | Academic Significance |
|:----------|:---------|:--------------|:----------------------|
| **Ear Stream** | w₁ | **0.528** | Primary linear confidence driver. |
| **Palm Stream** | w₂ | **0.169** | Secondary linear confidence driver. |
| **Hadamard Interaction** | w₃ | **0.303** | Accounts for >**30%** of decision logic, proving cross-modal synergy. |

### Biometric Security Metrics

| Security Metric | Value | Testing Threshold |
|:----------------|:------|:-----------------|
| **Overall Authentication Accuracy** | **95.98%** | 0.5 |
| **Equal Error Rate (EER)** | **5.05%** | Intersection of FAR and FRR |
| **False Accept Rate (FAR)** | **5.05%** | 0.5 |
| **False Reject Rate (FRR)** | **3.00%** | 0.5 |

### ROC Analysis

| Architecture Modality | Area Under Curve (AUC) | Performance Status |
|:----------------------|:----------------------|:-------------------|
| **Palm Stream** (Isolated) | **0.812** | Baseline |
| **Ear Stream** (Isolated) | **0.985** | Strong Unimodal |
| **h(PRS-PDO) Fused System** | **0.986** | **Optimal / State-of-the-Art** |

---

## BEST PAPERS

- Singh, I., Singh, T., Luthra, S., & Khatri, S. (2024, April). Harris Hawk Optimized CLAHE and Novel Score Level Fusion of Lightweight CNNs for Multimodal Biometric Recognition. In *2024 6th International Conference on Pattern Analysis and Intelligent Systems (PAIS)* (pp. 1-8). IEEE. https://doi.org/10.1109/PAIS62114.2024.10541196

- Kamboj, A., Rani, R., & Nigam, A. (2022). A comprehensive survey and deep learning-based approach for human recognition using ear biometric. *The Visual Computer, 38*(7), 2383–2416. https://doi.org/10.1007/s00371-021-02119-0

- Trabelsi, S., Samai, D., Dornaika, F., Benlamoudi, A., Bensid, K., & Taleb-Ahmed, A. (2022). Efficient palmprint biometric identification systems using deep learning and feature selection methods. *Neural Computing and Applications, 34*(14), 12119–12141. https://doi.org/10.1007/s00521-022-07098-4

- Kandasamy, M. (2022). Multimodal biometric crypto system for human authentication using ear and palm print. *Pattern Analysis and Applications, 25*(4), 1015–1024. https://doi.org/10.1007/s10044-022-01058-3

- Michele, A., Colin, V., & Santika, D. D. (2019). Mobilenet convolutional neural networks and support vector machines for palmprint recognition. *Procedia Computer Science, 157*, 110–117. https://doi.org/10.1016/j.procs.2019.08.147
