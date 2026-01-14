# Random Forest Models for Materials Discovery

## What is Random Forest?

Random Forest (RF) is an **ensemble learning method** that trains multiple decision trees and combines their predictions. For materials discovery, RF provides:

1. **Predictions**: Mean of all tree predictions
2. **Uncertainty**: Variance across tree predictions
3. **Robustness**: Resilience to outliers and noise

```
┌─────────────────────────────────────────────────────────────────┐
│                    Random Forest Ensemble                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌────────┐  ┌────────┐  ┌────────┐      ┌────────┐          │
│    │ Tree 1 │  │ Tree 2 │  │ Tree 3 │ ...  │Tree 100│          │
│    └───┬────┘  └───┬────┘  └───┬────┘      └───┬────┘          │
│        │           │           │               │                │
│        ▼           ▼           ▼               ▼                │
│      45 MPa      47 MPa      43 MPa    ...   46 MPa            │
│        │           │           │               │                │
│        └───────────┴───────────┴───────────────┘                │
│                          ↓                                       │
│              Mean = 45.25 MPa (Prediction)                       │
│              Std  = 1.58 MPa  (Uncertainty)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two Implementations: Scikit-learn RF vs LoLoPy RF

We provide **two** Random Forest implementations with key differences:

| Aspect | Scikit-learn RF | LoLoPy RF |
|--------|-----------------|-----------|
| **Library** | `sklearn.ensemble.RandomForestRegressor` | `lolopy.learners.RandomForestRegressor` |
| **Multi-target** | Native multi-output | Separate model per target |
| **Scaling** | StandardScaler (normalized) | No scaling (raw values) |
| **Uncertainty** | Tree variance (post-hoc) | **Built-in** calibrated uncertainty |
| **Speed** | Faster | Slightly slower |
| **Memory** | More efficient | More models (one per target) |

---

## Scikit-learn Random Forest (RFModel)

### Architecture

```python
# RFModel Structure
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  StandardScaler (X)                  │
│         ↓                            │
│  [Tree 1] [Tree 2] ... [Tree 100]    │
│         ↓                            │
│  Mean Prediction + Variance          │
│         ↓                            │
│  Inverse Transform (StandardScaler)  │
│         ↓                            │
│  Output: Predictions + Uncertainty    │
└──────────────────────────────────────┘
```

### How Uncertainty is Calculated

```python
# Each tree makes a prediction
tree_predictions = [tree.predict(X) for tree in model.estimators_]

# Uncertainty = variance across trees
variance = np.var(tree_predictions, axis=0)
uncertainty = np.sqrt(variance) * scaler_y.scale_  # Back to original scale
```

**Why variance works**: Trees trained on bootstrap samples see different data, producing different predictions. High variance = regions the forest is uncertain about.

### Key Files

| File | Class | Function |
|------|-------|----------|
| `app/models/rf_model.py` | `RFModel` | Core model wrapper |
| `app/models/rf_model.py` | `RFSurrogate` | SurrogateModel interface |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of trees |
| `random_state` | 42 | Reproducibility seed |

---

## LoLoPy Random Forest (LolopyRFModel)

### What is LoLoPy?

**LoLoPy** (Leave-One-out Learning of Optimal Predictions, yes!) is a library developed by **Citrine Informatics** specifically for materials informatics. Its key advantage:

> **Built-in calibrated uncertainty estimates** using leave-one-out cross-validation

Unlike scikit-learn RF which estimates uncertainty from tree variance, LoLoPy provides **calibrated** prediction intervals that better reflect the true model uncertainty.

### Architecture

```python
# LolopyRFModel Structure
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  ┌──────────────────────────────┐    │
│  │  Target 1: Lolopy RF Model   │    │
│  │  Target 2: Lolopy RF Model   │    │
│  │  ...                         │    │
│  │  Target N: Lolopy RF Model   │    │
│  └──────────────────────────────┘    │
│         ↓                            │
│  Predictions + Calibrated Uncertainty │
└──────────────────────────────────────┘
```

### How Uncertainty is Calculated

```python
# LoLoPy provides uncertainty directly via return_std=True
predictions, uncertainties = model.predict(X, return_std=True)
```

**Why it's better**: LoLoPy uses an internal cross-validation strategy that produces uncertainty estimates that are **properly calibrated** to the prediction error. This means:
- 68% of true values fall within ±1σ
- 95% fall within ±2σ

This is crucial for Bayesian Optimization!

### Key Files

| File | Class | Function |
|------|-------|----------|
| `app/models/lolopy_model.py` | `LolopyRFModel` | Core model wrapper |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_trees` | 100 | Number of trees per target |

---

## Do They Produce Different Results?

**Yes, they can produce meaningfully different results:**

### 1. **Uncertainty Estimates Differ**

| Model | Uncertainty Method | Quality |
|-------|-------------------|---------|
| Scikit-learn RF | Tree variance | Uncalibrated |
| LoLoPy RF | Leave-one-out CV | Calibrated |

This directly impacts the **acquisition function** (WEBSLAMD utility), which uses uncertainty for exploration:

```
Utility = (1-α)×Exploitation + α×Exploration
                                ───────────
                                Uncertainty
```

Different uncertainty estimates → Different sample selections!

### 2. **Scaling Behavior**

- **Scikit-learn RF**: Uses StandardScaler (mean=0, std=1)
- **LoLoPy RF**: No scaling (works with raw values)

This affects how the model handles features with different scales.

### 3. **Multi-Target Handling**

- **Scikit-learn RF**: Single model, multi-output
- **LoLoPy RF**: Separate model per target (more flexible, but independent)

---

## When to Use Which?

| Use Case | Recommended Model |
|----------|-------------------|
| **Small datasets (<100 samples)** | LoLoPy RF (better calibration) |
| **Large datasets (>1000 samples)** | Scikit-learn RF (faster) |
| **Well-calibrated uncertainty needed** | LoLoPy RF |
| **Quick experimentation** | Scikit-learn RF |
| **Multi-objective optimization** | Either (both support multi-target) |

---

## Comparison with Other Models

| Model | Strengths | RF Advantage |
|-------|-----------|--------------|
| **Gaussian Process** | Elegant uncertainty | RF is faster, scales better |
| **PINN** | Physics-informed | RF needs no domain knowledge |
| **Deep Kernel Learning** | Powerful + GP | RF is simpler to tune |
| **RL (PPO)** | Learns strategy | RF gives instant predictions |

---

## How RF Fits in the WEBSLAMD Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEBSLAMD Active Learning                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Train RF model on labeled data                              │
│         ↓                                                        │
│  2. Predict unlabeled candidates + get uncertainty              │
│         ↓                                                        │
│  3. Calculate WEBSLAMD Utility:                                  │
│     Utility = (1-α)×Exploitation + α×Exploration                │
│         ↓                                                        │
│  4. Select highest utility sample for testing                   │
│         ↓                                                        │
│  5. Get lab results, add to training data                       │
│         ↓                                                        │
│  6. Repeat from step 1                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Practical Tips

1. **Start with LoLoPy RF** for materials discovery - better uncertainty calibration
2. **Use Scikit-learn RF** if you need speed or have many samples
3. **Check uncertainty values** - if all uncertainties are similar, exploration is ineffective
4. **Monitor diversity** - RF should select diverse candidates, not cluster in one region
5. **Combine with LLM** in Hybrid mode for domain-informed exploration

---

## Technical Details

**Scikit-learn RF File**: `app/models/rf_model.py`

**Key Classes**:
- `RFModel`: Wrapper for RandomForestRegressor
- `RFSurrogate`: Implements SurrogateModel interface

**Key Functions**:
- `train_rf_model()`: Trains the model
- `evaluate_rf_model()`: Scores candidates with WEBSLAMD

---

**LoLoPy RF File**: `app/models/lolopy_model.py`

**Key Classes**:
- `LolopyRFModel`: Wrapper for lolopy RandomForestRegressor

**Key Functions**:
- `train_lolopy_model()`: Trains one model per target
- `evaluate_lolopy_model()`: Scores candidates with WEBSLAMD

---

## Installation Notes

**Scikit-learn RF**: Included with scikit-learn (already installed)

**LoLoPy RF**: Requires separate installation:
```bash
pip install lolopy
```

LoLoPy requires at least **8 training samples** to function properly (due to internal cross-validation).
