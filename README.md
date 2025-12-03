# Ratio-Guided Bimodal Flow Matching

Implementation of **ratio-guided bimodal Flow Matching** for generating coherent pairs of images (x, y).

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
    │   Train on x₁                        Train on y₁=T(x₁) │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                                ↓
    ┌──────────────────────────────────────────────────────────┐
    │              RATIO ESTIMATOR TRAINING                    │
    │                                                          │
    │   Real pairs: (x, T(x))  ──┐     ┌──  Fake pairs: (x, y')│
    │                            ↓     ↓                       │
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
    │       │    │                        │    │               │
    │       │    │  ∇_x log r̂(x_t, y_t)  │    │               │
    │       │    │  ∇_y log r̂(x_t, y_t)  │    │               │
    │       │    └────────────────────────┘    │               │
    │       │              ↑                   │               │
    │       └──────────────┼───────────────────┘               │
    │                      │                                   │
    │              ┌───────┴───────┐                          │
    │              │ r̂(x_t, y_t)  │                          │
    │              │   (frozen)    │                          │
    │              └───────────────┘                          │
    │                                                          │
    │       ▼                                  ▼               │
    │      x₁                                 y₁              │
    │   (generated)                       (generated)         │
    │                                                          │
    │            ══════════════════════════                   │
    │              COHERENT PAIR (x₁, y₁)                     │
    │            ══════════════════════════                   │
    └──────────────────────────────────────────────────────────┘

    ODE avec guidance:
    ┌────────────────────────────────────────────────────────┐
    │  dx_t/dt = v_x(x_t, t) + γ · ∇_x log r̂(x_t, y_t)     │
    │  dy_t/dt = v_y(y_t, t) + γ · ∇_y log r̂(x_t, y_t)     │
    └────────────────────────────────────────────────────────┘
```

## Key Features

- **Rectified Flow**: Noise-free CFM with constant velocity (`u_t = x_1 - x_0`)
- **Terminal Ratio**: Estimates `r̂_1(x,y)` at t=1 only (simpler, sufficient)
- **Stable Guidance**: Gradient clipping + warmup (t > 0.3)
- **Modular Design**: Easy to swap ratio estimators and guidance methods

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Classifier (for evaluation)

```bash
python src/train_classifier.py --epochs 3
```

### 2. Train Flow Matching Models

Train two independent FM models (standard and transformed MNIST):

```bash
# FM_x: standard MNIST
python src/train_flow.py --modality x --epochs 50

# FM_y: rotated MNIST
python src/train_flow.py --modality y --transform_type rotate90 --epochs 50
```

**Supported transformations:**
- `rotate90`, `rotate180`, `rotate270`
- `invert` (color inversion)
- `flip_h`, `flip_v` (horizontal/vertical flip)

### 3. Train Ratio Estimator

```bash
# Discriminator (default)
python src/train_ratio.py --loss_type disc --transform_type rotate90 --epochs 30

# RuLSIF (optional)
python src/train_ratio.py --loss_type rulsif --transform_type rotate90 --epochs 30
```

### 4. Sample Pairs

**Baseline (independent flow):**
```bash
python src/sample.py --transform_type rotate90 --guidance_method none --num_samples 64
```

**Ratio-guided:**
```bash
python src/sample.py \
    --transform_type rotate90 \
    --guidance_method grad_log_ratio \
    --guidance_strength 2.0 \
    --num_samples 64
```

### 5. Evaluate Coherence

```bash
python src/evaluate.py \
    --transform_type rotate90 \
    --guidance_methods none grad_log_ratio \
    --guidance_strengths 0.0 1.0 2.0 3.0 \
    --num_samples 1000
```

## Expected Results

| Method          | γ   | Coherence Accuracy |
|-----------------|-----|--------------------|
| Independent     | 0.0 | ~10% (chance)      |
| Ratio-guided    | 1.0 | ~60-70%            |
| Ratio-guided    | 2.0 | ~80-90%            |
| Ratio-guided    | 3.0 | ~85-95%            |

**Note:** Higher γ improves coherence but may degrade marginal quality.

## Project Structure

```
ratioGuidedFM/
├── src/
│   ├── models/
│   │   ├── flow_matching.py          # FlowMatchingModel (CFM)
│   │   ├── ratio_estimator.py        # RatioEstimator
│   │   └── classifier.py             # MNIST classifier
│   ├── data/
│   │   ├── transforms.py             # Image transformations
│   │   └── mnist_dataset.py          # Dataset loaders
│   ├── utils/
│   │   ├── flow_utils.py             # CFM utilities, sampling
│   │   ├── losses.py                 # Ratio losses
│   │   ├── trainer.py                # RatioTrainer
│   │   └── path_utils.py             # Checkpoint management
│   ├── train_flow.py                 # Train FM models
│   ├── train_ratio.py                # Train ratio estimator
│   ├── train_classifier.py           # Train classifier
│   ├── sample.py                     # Generate samples
│   └── evaluate.py                   # Evaluate coherence
├── checkpoints/                      # Saved models
├── outputs/                          # Generated samples
└── README.md
```

## Mathematical Details

### Flow Matching (Rectified Flow)

**Interpolation:**
```
x_t = (1-t) * x_0 + t * x_1
```

**Velocity field:**
```
u_t(x_t | x_0, x_1) = x_1 - x_0
```

**Training objective:**
```
L = E_{t, x_0, x_1, x_t} ||v_θ(x_t, t) - (x_1 - x_0)||²
```

### Ratio Estimation (Discriminator)

**Optimal discriminator:**
```
D*(z) = q(z) / (q(z) + p(z))
```

**Log-ratio:**
```
log r̂(z) = log(D/(1-D)) = F.logsigmoid(T) - F.logsigmoid(-T)
```

### Guided Sampling

**ODE with guidance:**
```
dx_t/dt = u_t^x(x_t, t) + γ · ∂_x log r̂_1(x_t, y_t)
dy_t/dt = u_t^y(y_t, t) + γ · ∂_y log r̂_1(x_t, y_t)
```

**Stability features:**
- Gradient clipping: `||∇ log r̂|| ≤ 1`
- Warmup: guidance only for `t > 0.3`

## Alternative Approach (Simpler Baseline)

Instead of ratio guidance, you could train a **single joint FM** directly on paired data:

```bash
# Hypothetical joint FM (not implemented)
x_joint_FM(concat(x,y), t) → [v_x, v_y]
```

**Pros:**
- Simpler (one model instead of three)
- Guaranteed coherence

**Cons:**
- Loses modularity (can't reuse pretrained unimodal FMs)
- Doesn't demonstrate ratio guidance methodology

We chose the ratio guidance approach to showcase the technique and maintain flexibility.

## References

- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (2023)
- **Ratio Guidance**: Inspired by [ratio-guidance-light](https://github.com/foubari/ratio-guidance-light) (diffusion-based)
- **Rectified Flow**: Liu et al., "Flow Straight and Fast" (2022)

## Citation

If you use this code, please cite:

```bibtex
@software{ratio_guided_fm_2025,
  title={Ratio-Guided Bimodal Flow Matching},
  author={Fouad Oubari},
  year={2025},
  url={TODO: link will be added}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
