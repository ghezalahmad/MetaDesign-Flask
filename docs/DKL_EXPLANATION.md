# Deep Kernel Learning (DKL) for Materials Discovery

## What is Deep Kernel Learning?

Deep Kernel Learning (DKL) combines the power of **Deep Neural Networks** with **Gaussian Processes** (GPs). It uses a neural network to learn rich feature representations, then applies GPs on these features for predictions with principled uncertainty.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DKL Architecture                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Raw Input Features                                           │
│           ↓                                                      │
│    ┌─────────────────┐                                          │
│    │   Deep Neural   │  ← Learns complex feature transformations │
│    │    Network      │                                          │
│    │  (2 layers)     │                                          │
│    └────────┬────────┘                                          │
│             ↓                                                    │
│    Learned Feature Space                                        │
│             ↓                                                    │
│    ┌─────────────────┐                                          │
│    │   Gaussian      │  ← Provides calibrated uncertainty        │
│    │   Process       │                                          │
│    └────────┬────────┘                                          │
│             ↓                                                    │
│    Predictions + Uncertainty                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The neural network learns a better kernel space where the GP can model complex relationships, while the GP provides principled Bayesian uncertainty estimates.

---

## Why Combine NN + GP?

| Component | Strength | Weakness |
|-----------|----------|----------|
| **Neural Networks** | Learn complex features | Overconfident, no uncertainty |
| **Gaussian Processes** | Beautiful uncertainty | Limited to simple kernels |
| **DKL (Combined)** | Both! | More computation |

---

## Our Implementation

### Architecture

```python
# FeatureExtractor (Neural Network)
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  Hidden Layer 1 (64 neurons + ReLU)  │
│         ↓                            │
│  Dropout (0.2)                       │
│         ↓                            │
│  Hidden Layer 2 (64 neurons + ReLU)  │
│         ↓                            │
│  Output: Feature Dimension           │
└──────────────────────────────────────┘
         ↓
┌──────────────────────────────────────┐
│  Independent GPs (one per target)     │
│                                       │
│  GP₁ → Target 1 prediction + σ₁      │
│  GP₂ → Target 2 prediction + σ₂      │
│  ...                                  │
└──────────────────────────────────────┘
```

### Training Process

**Step 1**: Train the Neural Network
```python
# NN learns to map inputs → target-like features
for epoch in range(epochs):
    features = feature_extractor(X)
    loss = MSE(features, y_scaled)  # Proxy loss
    loss.backward()
```

**Step 2**: Extract Features and Fit GPs
```python
# Transform inputs through trained NN
X_features = feature_extractor(X_tensor)

# Fit one GP per target in feature space
for i, target in enumerate(target_columns):
    gp.fit(X_features, y[:, i])
```

### Key Files

| File | Class | Function |
|------|-------|----------|
| `app/models/dkl_surrogate_model.py` | `DKLModel` | Main DKL implementation |
| `app/models/dkl_surrogate_model.py` | `FeatureExtractor` | Neural network component |

---

## Kernel Configuration

The GP uses a composite kernel:

```
Kernel = C × RBF + WhiteKernel
```

| Component | Purpose |
|-----------|---------|
| `ConstantKernel (C)` | Scales output variance |
| `RBF` | Radial Basis Function for smoothness |
| `WhiteKernel` | Models observation noise |

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 64 | Neurons per hidden layer |
| `epochs` | 100 | NN training epochs |
| `lr` | 0.001 | Learning rate |
| `alpha` | 1e-6 | GP regularization |

---

## When to Use DKL

| Use Case | Recommended |
|----------|-------------|
| **Complex feature interactions** | ✅ DKL excels |
| **Need calibrated uncertainty** | ✅ GP provides this |
| **Small datasets (<50 samples)** | ⚠️ Consider simpler GP |
| **Large datasets (>1000)** | ✅ NN helps scale |

---

## Comparison with Other Models

| Model | Uncertainty | Feature Learning | Scalability |
|-------|-------------|------------------|-------------|
| **Pure GP** | ✅ Excellent | ❌ Fixed kernel | ❌ O(n³) |
| **Random Forest** | ⚠️ Heuristic | ❌ Decision trees | ✅ Good |
| **PINN** | ⚠️ MC Dropout | ✅ Physics-informed | ✅ Good |
| **DKL** | ✅ GP-based | ✅ Neural network | ✅ Good |
