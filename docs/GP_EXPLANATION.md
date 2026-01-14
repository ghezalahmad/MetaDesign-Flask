# Gaussian Process (GP) Model for Materials Discovery

## What is a Gaussian Process?

A Gaussian Process (GP) is a probabilistic model that defines a distribution over functions. Unlike neural networks that output point estimates, GPs provide:

1. **Predictions**: Mean function value
2. **Uncertainty**: Confidence intervals around predictions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gaussian Process Intuition                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Training Points (●)                                          │
│                      ●                                          │
│                 ●        ●                                      │
│            ●                  ●                                 │
│                                                                  │
│    GP fits a smooth function with uncertainty bands:            │
│                                                                  │
│         ╭~~~~~~~~~~~~~~~╮  ← Wide uncertainty (no data)         │
│       ●══════════●══════════●  ← Narrow at training points     │
│         ╰~~~~~~~~~~~~~~~╯                                       │
│                                                                  │
│    "Where I have data, I'm confident."                          │
│    "Where I don't have data, I'm uncertain."                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why GP for Materials Discovery?

| Advantage | Description |
|-----------|-------------|
| **Principled uncertainty** | Bayesian probability, not heuristics |
| **Data-efficient** | Works well with small datasets |
| **Smooth interpolation** | Physical properties vary smoothly |
| **Automatic acquisition** | Uncertainty drives exploration |

---

## Our Implementation

### Architecture

```python
# GPModel Structure (Multi-Output via Independent GPs)
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  StandardScaler (normalize X)        │
│         ↓                            │
│  ┌─────────────────────────────────┐ │
│  │  GP₁ → Target 1 + σ₁            │ │
│  │  GP₂ → Target 2 + σ₂            │ │
│  │  ...                            │ │
│  │  GPₙ → Target N + σₙ            │ │
│  └─────────────────────────────────┘ │
│         ↓                            │
│  Inverse Transform (StandardScaler)  │
│         ↓                            │
│  Output: Predictions + Uncertainty   │
└──────────────────────────────────────┘
```

### Kernel Configuration

The default kernel combines three components:

```
Kernel = C × RBF + WhiteKernel
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| **ConstantKernel (C)** | `C(1.0)` | Scales output variance |
| **RBF (Radial Basis Function)** | `RBF(1.0)` | Smoothness/similarity |
| **WhiteKernel** | `WhiteKernel(α)` | Observation noise |

```python
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(α)
```

Bounds allow automatic hyperparameter optimization!

---

## Training Process

```python
# 1. Scale inputs and targets
X_scaled = scaler_x.fit_transform(X)
Y_scaled = scaler_y.fit_transform(Y)

# 2. Fit one GP per target
for i in range(n_targets):
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10  # Find good hyperparameters
    )
    gp.fit(X_scaled, Y_scaled[:, i])
```

The `n_restarts_optimizer=10` is important—it helps find optimal kernel hyperparameters.

---

## Uncertainty Estimation

GP provides **closed-form** uncertainty (no sampling needed):

```python
# GP gives mean and std directly
mean, std = gp.predict(X, return_std=True)

# Inverse transform to original scale
uncertainty = std * scaler_y.scale_
```

This is more theoretically grounded than MC Dropout used in neural networks.

---

## Key Files

| File | Class | Function |
|------|-------|----------|
| `app/models/gp_model.py` | `GPModel` | Core GP wrapper |
| `app/models/gp_model.py` | `GPSurrogate` | SurrogateModel interface |

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 1e-10 | Noise level (regularization) |
| `normalize_y` | True | Normalize targets during fitting |
| `n_restarts_optimizer` | 10 | Kernel hyperparameter search restarts |
| `random_state` | 42 | Reproducibility seed |

### Tuning Tips

- **↑ alpha**: More regularization, smoother predictions
- **↓ alpha**: Tighter fit to data, may overfit
- **↑ n_restarts**: Better hyperparameters, slower training

---

## Scalability Limitation

⚠️ **GPs have O(n³) complexity** for `n` training samples:

| Samples | Relative Time |
|---------|---------------|
| 100 | 1× |
| 500 | 125× |
| 1000 | 1000× |

For large datasets (>500 samples), consider:
- **DKL**: Neural network reduces dimensionality before GP
- **Random Forest**: O(n log n) complexity
- **PINN**: Scales linearly with data

---

## When to Use GP

| Scenario | Recommendation |
|----------|----------------|
| **Small dataset (<100 samples)** | ✅ GP excels |
| **Need calibrated uncertainty** | ✅ Best choice |
| **Smooth physical properties** | ✅ Ideal |
| **Large dataset (>500)** | ⚠️ Consider RF or DKL |
| **Complex feature interactions** | ⚠️ Consider DKL or PINN |

---

## Comparison with Other Models

| Model | Uncertainty | Scalability | Feature Learning |
|-------|-------------|-------------|------------------|
| **GP** | ✅ Closed-form | ❌ O(n³) | ❌ Fixed kernel |
| **Random Forest** | ⚠️ Tree variance | ✅ Good | ❌ Decision trees |
| **DKL** | ✅ GP-based | ✅ NN helps | ✅ Neural network |
| **PINN** | ⚠️ MC Dropout | ✅ Linear | ✅ Physics-informed |

---

## Practical Tips

1. **Use GP for small datasets** (<100 samples) where you need reliable uncertainty
2. **Check kernel fit** - badly tuned kernels give poor uncertainty
3. **Monitor training time** - if slow, switch to RF or DKL
4. **StandardScaler is important** - GP assumes normalized data
5. **Consider GP as baseline** - compare other models against it
