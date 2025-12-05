# Ratio-Guided Multimodal Flow Matching

Implementation of **ratio-guided multimodal Flow Matching** for generating coherent pairs of images across different modalities.

## Experiments

### 1. MNIST Transforms (Same Modality)
Generate pairs (MNIST, Rotated-MNIST) where both show the same digit.

### 2. MNIST-SVHN (Cross-Modality) 
Generate pairs (MNIST, SVHN) where both represent the same digit class.
- MNIST: 1×32×32 grayscale
- SVHN: 3×32×32 RGB

## Architecture

```
                           TRAINING PHASE
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   ┌─────────────┐                    ┌─────────────┐    │
    │   │   FM_x      │                    │   FM_y      │    │
    │   │  (prior→x)  │                    │  (prior→y)  │    │
    │   └─────────────┘                    └─────────────┘    │
    │         ↓                                  ↓            │
    │   Train on x₁                        Train on y₁        │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                                ↓
    ┌──────────────────────────────────────────────────────────┐
    │              RATIO ESTIMATOR TRAINING                    │
    │                                                          │
    │   Real pairs: (x, y) same class ──┐  ┌── Fake: different│
    │                                   ↓  ↓                   │
    │                     ┌─────────────────┐                  │
    │                     │  r̂(x,y) = q/p  │                  │
    │                     │  Discriminator  │                  │
    │                     └─────────────────┘                  │
    └──────────────────────────────────────────────────────────┘

                          SAMPLING PHASE
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   x₀ ~ N(0,I)                        y₀ ~ N(0,I)        │
    │       │                                  │               │
    │       ▼                                  ▼               │
    │   ┌───────┐                          ┌───────┐          │
    │   │ FM_x  │◄─────────┐    ┌─────────►│ FM_y  │          │
    │   └───┬───┘          │    │          └───┬───┘          │
    │       │              │    │              │               │
    │       │    ┌─────────┴────┴─────────┐    │               │
    │       │    │    RATIO GUIDANCE      │    │               │
    │       │    │  ∇_x log r̂(x_t, y_t)  │    │               │
    │       │    │  ∇_y log r̂(x_t, y_t)  │    │               │
    │       │    └────────────────────────┘    │               │
    │       ▼                                  ▼               │
    │      x₁                                 y₁              │
    │            ══════════════════════════                   │
    │              COHERENT PAIR (x₁, y₁)                     │
    │            ══════════════════════════                   │
    └──────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Experiment 1: MNIST Transforms

```bash
# 1. Train classifier (for evaluation)
python src/train_classifier.py --epochs 3

# 2. Train flow matching models
python src/train_flow.py --modality x --epochs 50
python src/train_flow.py --modality y --transform_type rotate90 --epochs 50

# 3. Train ratio estimator
python src/train_ratio.py --loss_type disc --transform_type rotate90 --epochs 30

# 4. Sample with guidance
python src/sample.py --transform_type rotate90 --guidance_method mc_feng --guidance_strength 0.5

# 5. Evaluate
python src/evaluate.py --transform_type rotate90 --guidance_strengths 0.0 0.5 1.0 2.0
```

### Experiment 2: MNIST-SVHN

```bash
# 1. Train classifiers
python src/train_classifiers_mnist_svhn.py --epochs 10

# 2. Train flow matching models
python src/train_flow_mnist32.py --epochs 50
python src/train_flow_svhn.py --epochs 50

# 3. Train ratio estimator (3.3M params, asymmetric encoders)
python src/train_ratio_mnist_svhn.py --epochs 30

# 4. Sample with guidance
python src/sample_mnist_svhn.py --guidance_method mc_feng --guidance_strength 0.5

# 5. Evaluate
python src/evaluate_mnist_svhn.py --guidance_strengths 0.0 0.5 1.0 2.0 5.0
```

## Project Structure

```
ratioGuidedFM/
├── src/
│   ├── models/
│   │   ├── flow_matching.py          # FM for MNIST 28x28
│   │   ├── unet.py                   # U-Net FM for MNIST
│   │   ├── unet_flexible.py          # U-Net FM for MNIST32/SVHN
│   │   ├── ratio_estimator.py        # Ratio for same-modality
│   │   ├── ratio_flexible.py         # Ratio for cross-modality (3.3M)
│   │   ├── classifier.py             # MNIST 28x28 classifier
│   │   └── svhn_classifier.py        # MNIST32 & SVHN classifiers
│   ├── data/
│   │   ├── mnist_dataset.py          # MNIST + transforms
│   │   └── mnist_svhn_dataset.py     # MNIST-SVHN paired dataset
│   ├── utils/
│   │   ├── flow_utils.py             # Sampling with guidance
│   │   └── losses.py                 # Discriminator/RuLSIF losses
│   │
│   ├── train_flow.py                 # Train FM (MNIST transforms)
│   ├── train_flow_mnist32.py         # Train FM for MNIST 32x32
│   ├── train_flow_svhn.py            # Train FM for SVHN
│   ├── train_ratio.py                # Train ratio (same modality)
│   ├── train_ratio_mnist_svhn.py     # Train ratio (cross-modality)
│   ├── sample.py                     # Sample MNIST transforms
│   ├── sample_mnist_svhn.py          # Sample MNIST-SVHN pairs
│   ├── evaluate.py                   # Evaluate MNIST transforms
│   └── evaluate_mnist_svhn.py        # Evaluate MNIST-SVHN
│
├── checkpoints/                      # Saved models
├── outputs/
│   ├── mnist_transform/              # MNIST transform results
│   └── mnist_svhn/                   # MNIST-SVHN results
└── README.md
```

## Guidance Methods

### MC Feng (Monte Carlo)
Generates independent samples from both flows, then uses the ratio estimator to weight them via importance sampling.

```python
# Weighted velocity
v_guided = (1 - γ) * v_ind + γ * Σ w_i * v_cond_i
```

### Gradient Log-Ratio
Adds gradient of log-ratio directly to the velocity field.

```python
v_guided = v_ind + γ * ∇ log r̂(x_t, y_t)
```

## Model Sizes

| Model | Parameters | Description |
|-------|------------|-------------|
| FM_mnist (U-Net) | ~460K | MNIST 28x28 or 32x32 |
| FM_svhn (U-Net) | ~6M | SVHN 32x32 RGB |
| Ratio (same modality) | ~944K | MNIST transforms |
| Ratio (cross-modality) | ~3.3M | MNIST-SVHN (asymmetric encoders) |

## References

- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (2023)
- **Rectified Flow**: Liu et al., "Flow Straight and Fast" (2022)

## License

MIT License
