# Edge-Optimized Few-Shot Multimodal Biometric Authentication via Chiral Siamese Networks and h(PRS-PDO) Fusion

**Target Venue:** IEEE International Conference on Consumer Electronics (ICCE)
**Application Domain:** Edge AI, Biometric Security, Few-Shot Learning (FSL)

---

## 1. Core Innovations & Contributions
This architecture addresses the severe computational, memory, and data constraints of consumer edge devices (e.g., smartphones, smart locks) by introducing four highly novel mechanisms:

1. **The Chiral Mirror Mechanism (Zero-Memory FSL):** Overcomes extreme few-shot limitations (4 enrollment images per user) by exploiting anatomical symmetry. Horizontally flipping the left palm to mathematically mirror the right hand doubles the effective support set and enables Siamese weight-sharing without storing additional data.
2. **Topographic Peak Maps:** Replaces battery-draining $O(N^2)$ spatial denoising algorithms with a linear-time $O(N/s^2)$ Fast Guided Filter (FGF) pipeline, utilizing $255 - I$ inversion to cast biometric structures as high-intensity activation peaks for lightweight neural networks.
3. **Dual-Stream Ghost-MobileNetV3 with Coordinate Attention:** A heterogeneous extractor featuring Spatial Dropout ($p=0.4$) and Coordinate Attention (CA) to independently encode 1D $X/Y$ spatial coordinates, forcing the network to learn universal geometry rather than memorizing enrollment pixels.
4. **Single-Agent h(PRS-PDO) Non-Linear Fusion:** Replaces memory-heavy cross-attention modules with a lightweight metaheuristic optimizer. It balances unimodal confidence scores via a non-linear Hadamard interaction term, proving that the cross-correlation of modalities yields superior security.

---

## 2. Proposed Methodology

The pipeline is divided into three sequential hardware-aware phases:

### Phase 1: Hardware-Aware Preprocessing (Data Amplification)
* **Dataset Marshalling:** The dataset is rigorously partitioned using Stratified Balanced Sampling (2:1 Train/Test split independently for native right and native left hands) to prevent modality-source bias.
* **Combinatorics:** Using Siamese mathematics, the 4 enrollment images are expanded into $\binom{4}{2} = 6$ positive pairs per user, scaling 100 subjects into 996 unique FSL training scenarios.
* **Topographic Pipeline:** Raw arrays undergo continuous-tone inversion, Localized CLAHE ($\beta=2.0$), LS-AMF median approximation, and an FGF downsampling ($s=4$) to isolate structural cartilage and principal lines.

### Phase 2: Heterogeneous Feature Extraction
* **Stream A (Ear):** Processes continuous-tone biometric maps directly through the modified Ghost-MobileNetV3 backbone.
* **Stream B (Chiral Siamese Palm):** Processes binary biometric maps. To combat overfitting on the microscopic dataset, the network utilizes in-memory zero-byte geometric augmentations (±5° rotation, 10% distortion) and tightened L2 Contrastive Loss limits (margin $m=1.0$).
* Both streams culminate in a Coordinate Attention block and yield normalized embedding vectors.

### Phase 3: Score-Level Metaheuristic Fusion
Rather than concatenating feature vectors (which bloats edge-device memory), the system fuses final confidence scores ($C_{ear}$ and $C_{palm}$). 
* **The Single-Agent Optimizer:** A hybrid agent driven by energy decay ($E$) transitions from global Prism Refraction Search (PRS) to local Prairie Dog Optimization (PDO).
* **The Objective Function:** The agent solves for three weights to minimize classification error:
  $$C_{fused} = w_1 \cdot C_{ear} + w_2 \cdot C_{palm} + w_3(C_{ear} \circ C_{palm})$$
* The $w_3$ Hadamard product term captures the hidden synergistic security value between the two modalities without requiring a secondary neural network.

---

## 3. Architectural Data Flow

```text
[Raw Input] --> Left Palm (Flip) + Right Palm --> [Siamese Pool: 4 Train / 2 Test]
                                                             |
                                                             v
[Ear Input] ------------------> [Topographic Preprocessing (255-I, CLAHE, FGF)]
                                                             |
                 +-------------------------------------------+-----------------------------------+
                 |                                                                               |
          [Stream A: Ear]                                                          [Stream B: Palm Pairs]
                 |                                                                               |
         (Ghost-MobileNetV3)                                                          (Ghost-MobileNetV3)
       (Coordinate Attention)                                                       (Coordinate Attention)
                 |                                                                               |
         [C_ear Score]                                                              [C_palm Score]
                 |                                                                               |
                 +----------------------------> [h(PRS-PDO) Agent] <-----------------------------+
                                                       |
                                    [w1*Ear] + [w2*Palm] + [w3*(Ear ∘ Palm)]
                                                       |
                                              [Decision Threshold]
