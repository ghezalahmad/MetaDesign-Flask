# Physics-Informed Neural Networks (PINN) for Materials Discovery

## What is a Physics-Informed Neural Network?

A Physics-Informed Neural Network (PINN) is a neural network that incorporates **physical laws** directly into its training process. Unlike standard neural networks that learn purely from data, PINNs combine:

1. **Data-Driven Learning**: Fitting measured experimental points
2. **Physics Constraints**: Enforcing known physical laws and relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    PINN Training Process                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────────┐        ┌─────────────┐                       │
│    │   Neural    │        │   Physics   │                       │
│    │   Network   │        │    Laws     │                       │
│    └──────┬──────┘        └──────┬──────┘                       │
│           │                       │                              │
│           ▼                       ▼                              │
│    ┌─────────────┐        ┌─────────────┐                       │
│    │  Data Loss  │        │ Physics Loss│                       │
│    │  (MSE fit)  │        │ (constraints)│                       │
│    └──────┬──────┘        └──────┬──────┘                       │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌───────────────┐                                   │
│              │  Total Loss   │                                   │
│              │ = Data + λ×Phy│                                   │
│              └───────────────┘                                   │
│                                                                  │
│  The network learns to fit data WHILE respecting physics!        │
└──────────────────────────────────────────────────────────────────┘
```

**Key Insight**: By encoding domain knowledge, PINNs can make better predictions with less data—critical when experiments are expensive.

---

## Our Implementation

### Architecture

```python
# PINNModel Architecture
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  Hidden Layer 1 (128 neurons + ReLU) │
│         ↓                            │
│  Dropout (0.3)                       │
│         ↓                            │
│  Hidden Layer 2 (128 neurons + ReLU) │
│         ↓                            │
│  Dropout (0.3)                       │
│         ↓                            │
│  Hidden Layer 3 (128 neurons + ReLU) │
│         ↓                            │
│  Dropout (0.3)                       │
│         ↓                            │
│  Output Layer (n_targets)            │
└──────────────────────────────────────┘
```

| Component | Purpose |
|-----------|---------|
| **ReLU Activation** | Non-linear transformations |
| **Dropout** | Regularization + MC Uncertainty |
| **RobustScaler** | Handles outliers in inputs/targets |

### File Structure

| File | Purpose |
|------|---------|
| `app/models/pinn_model.py` | Core PINN model and training logic |
| `app/pinn_utils.py` | Physics loss functions for various domains |

---

## How PINN Training Works

### Step 1: Data Preparation

```python
# Scale inputs and targets using RobustScaler
scaler_x = RobustScaler().fit(data[input_columns])
scaler_y = RobustScaler().fit(labeled_data[target_columns])
```

RobustScaler is preferred because materials data often contains outliers.

### Step 2: Loss Function

The total loss combines data fit and physics constraints:

```
Total Loss = Data Loss + λ × Physics Loss
```

Where:
- **Data Loss**: Mean Squared Error between predictions and measurements
- **Physics Loss**: Penalties for violating physical laws
- **λ (lambda)**: Physics weight (default: 0.1)

### Step 3: Physics Constraints

Our implementation applies **universal** and **domain-specific** constraints:

#### Universal Constraints (Applied to ALL Datasets)

| Constraint | What It Does |
|------------|--------------|
| **Non-negativity** | Physical quantities shouldn't be negative |
| **Smoothness** | Properties should vary smoothly with inputs |
| **Anti-collapse** | Prevents all predictions becoming identical |

#### Domain-Specific Constraints (Automatic Detection)

When the model detects specific columns, it applies relevant physics:

| Domain | Detected Columns | Physics Applied |
|--------|------------------|-----------------|
| **Concrete** | water, cement, slag | Abrams' law (↑water → ↓strength) |
| **Metals** | - | Yield strength bounds (50-2000 MPa) |
| **Polymers** | - | Tensile strength bounds (10-150 MPa) |
| **Batteries** | - | Capacity bounds (0-300 mAh/g) |

Example: If your data contains "water" and "cement" columns, the PINN automatically applies:
- Higher water content → Lower strength (Abrams' law)
- More binder → Higher strength constraint

---

## Uncertainty Estimation: MC Dropout

PINN uses **Monte Carlo Dropout** for uncertainty quantification:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MC Dropout Process                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each sample, run N forward passes with dropout enabled:     │
│                                                                  │
│    Pass 1: [0.85, 0.92]  ─┐                                     │
│    Pass 2: [0.83, 0.89]  ─┤                                     │
│    Pass 3: [0.87, 0.91]  ─┼──► Mean = [0.85, 0.91]              │
│    ...                   ─┤      Std = [0.02, 0.015]            │
│    Pass N: [0.84, 0.90]  ─┘                                     │
│                                                                  │
│  Uncertainty = Standard Deviation across passes                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Works**: Dropout randomly "turns off" neurons. Different dropout patterns create slightly different predictions. The spread of these predictions indicates model uncertainty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 30 | Number of MC forward passes |
| `dropout_rate` | 0.3 | Probability of dropping neurons |

---

## Why PINN is Good for Materials Discovery

### 1. **Works with Limited Data**

Materials experiments are expensive. PINNs leverage physics to:
- Extrapolate beyond training data
- Avoid physically impossible predictions
- Generalize from fewer examples

### 2. **Respects Physical Laws**

Standard ML might predict:
- Negative compressive strength ❌
- Increasing strength with more water ❌

PINN prevents these physically impossible predictions.

### 3. **Automatic Constraint Detection**

Our implementation automatically detects domain:
```python
# If your columns contain 'water', 'cement', etc.
# → Concrete physics applied automatically
```

No manual configuration needed!

### 4. **Uncertainty-Aware Optimization**

MC Dropout uncertainty feeds into the WEBSLAMD acquisition function:
```
Utility = (1-α)×Exploitation + α×Exploration
         ────────────────       ───────────
         Predicted Value        Uncertainty
```

---

## Comparison with Other Models

| Model | Strengths | When to Use PINN |
|-------|-----------|------------------|
| **Gaussian Process** | Elegant uncertainty | When physics knowledge exists |
| **Random Forest** | Fast, robust | When constraints matter |
| **Deep Kernel Learning** | Combines GP + NN | When you want stronger physics |
| **PINN** | Physics-informed | For materials with known laws |

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 128 | Neurons per hidden layer |
| `num_layers` | 3 | Number of hidden layers |
| `dropout_rate` | 0.3 | MC Dropout probability |
| `epochs` | 100 | Training iterations |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `physics_loss_weight` | 0.1 | Balance between data and physics |
| `batch_size` | 32 | Training batch size |

### Tuning Tips

- **↑ physics_loss_weight**: Stronger physics constraints, may underfit data
- **↓ physics_loss_weight**: Better data fit, may violate physics
- **↑ dropout_rate**: More uncertainty, more regularization
- **↑ epochs**: Better convergence (watch for overfitting)

---

## Technical Details

**File**: `app/models/pinn_model.py`

**Key Classes**:
- `PINNModel`: PyTorch neural network with MC Dropout
- `PINNSurrogate`: Wrapper implementing `SurrogateModel` interface

**Key Functions**:
- `pinn_train()`: Trains the PINN model with physics loss
- `evaluate_pinn()`: Scores candidates using WEBSLAMD utility
- `compute_physics_loss()`: Calculates adaptive physics constraints

**Physics Constraints File**: `app/pinn_utils.py`

**Domain-Specific Functions**:
- `_compute_concrete_physics()`: Concrete/cementitious constraints
- `_compute_metals_physics()`: Metal alloy constraints
- `_compute_polymer_physics()`: Polymer constraints
- `_compute_battery_physics()`: Battery material constraints

---

## Practical Tips

1. **Start with default settings** - they work well for most materials
2. **Use for materials with known physics** - PINN shines when domain knowledge exists
3. **Check physics_loss_weight** - if predictions violate physics, increase it
4. **Monitor training logs** - look for balanced Data Loss vs Physics Loss
5. **Combine with LLM** in Hybrid mode for best results
