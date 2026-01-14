# Meta-Learning Models: Reptile & ProtoNet

## What is Meta-Learning?

Meta-learning, or "learning to learn," trains models that can quickly adapt to new tasks with minimal data. This is perfect for materials discovery where:

- New material systems emerge frequently
- Experiments are expensive (limited data)
- Related problems share underlying patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                    Meta-Learning Paradigm                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Traditional ML:                                              │
│    ┌─────────────┐                                              │
│    │  One Task   │ → Train → Predict (same task only)           │
│    └─────────────┘                                              │
│                                                                  │
│    Meta-Learning:                                               │
│    ┌─────────────┐                                              │
│    │  Many Tasks │ → Learn to Adapt → Quick adaptation          │
│    │   (train)   │                    to NEW tasks              │
│    └─────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

# Reptile Model

## What is Reptile?

Reptile is a first-order meta-learning algorithm (simpler than MAML) that learns an initialization point from which the model can quickly adapt to new tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reptile Algorithm                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Initial Weights (θ₀)                                         │
│           ↓                                                      │
│    For each meta-epoch:                                          │
│    ┌─────────────────────────────────────────┐                  │
│    │  1. Sample a task (batch of data)        │                  │
│    │  2. Copy weights: θ_before = θ           │                  │
│    │  3. Train on task for K steps → θ_after  │                  │
│    │  4. Reptile update:                      │                  │
│    │     θ = θ_before + ε(θ_after - θ_before) │                  │
│    └─────────────────────────────────────────┘                  │
│           ↓                                                      │
│    Final Weights (good initialization for new tasks)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Idea

Instead of optimizing for one task, Reptile finds weights that are a **good starting point** for any task. The model learns general patterns that transfer across different material compositions.

## Our Implementation

### Architecture

```python
# ReptileModel Architecture  
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  LayerNorm (input normalization)     │
│         ↓                            │
│  Hidden Layer 1 (128 + LayerNorm)    │
│         ↓  ↘                         │
│  Hidden Layer 2 (128 + LayerNorm) + skip connection │
│         ↓  ↘                         │
│  Hidden Layer 3 (128 + LayerNorm) + skip connection │
│         ↓                            │
│  Output Layer → Predictions          │
└──────────────────────────────────────┘
```

Features:
- **LayerNorm**: Stabilizes training
- **Skip connections**: Enables deeper networks
- **Per-target models**: Trains separate model for each target (like LoLoPy)

### Training Algorithm

```python
for meta_epoch in range(epochs):
    # Decaying step size (0.1 → 0.01)
    reptile_step = 0.1 * (1 - 0.9 * meta_epoch / epochs)
    
    # Save current weights
    weights_before = model.state_dict()
    
    # Inner loop: 10 gradient steps on sampled task
    for _ in range(10):
        loss = MSE(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    
    # Reptile update: move towards adapted weights
    weights_after = model.state_dict()
    θ_new = θ_before + reptile_step × (θ_after - θ_before)
```

### Uncertainty Estimation

Uses **MC Dropout** with 50 forward passes:
```python
predictions = [model(X) for _ in range(50)]
mean = np.mean(predictions)
uncertainty = np.std(predictions)
```

---

# ProtoNet Model

## What is ProtoNet?

Prototypical Networks learn to create **prototypes** (representative embeddings) for each class/task. Predictions are made by comparing query samples to these prototypes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ProtoNet Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Support Set (few examples)                                   │
│           ↓                                                      │
│    ┌─────────────────┐                                          │
│    │    Encoder      │  ← Shared embedding network               │
│    │  (3 layers)     │                                          │
│    └────────┬────────┘                                          │
│             ↓                                                    │
│    [e₁, e₂, ..., eₖ]  (support embeddings)                      │
│             ↓                                                    │
│    Prototype = mean(support embeddings)                         │
│             ↓                                                    │
│    ┌─────────────────┐                                          │
│    │  Query Sample   │ → Encoder → Compare to prototype         │
│    └─────────────────┘                                          │
│             ↓                                                    │
│    Distance → Weighted prediction                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Idea

ProtoNet learns an embedding space where similar samples cluster together. The prototype (mean of support embeddings) represents the "center" of a task.

## Our Implementation

### Architecture

```python
# ProtoNetModel Encoder
┌──────────────────────────────────────┐
│  Input: Features (n_features)         │
│         ↓                            │
│  Layer 1 (embedding_size) + LayerNorm│
│         ↓  ↘                         │
│  Layer 2 (embedding_size) + LayerNorm + skip │
│         ↓  ↘                         │
│  Layer 3 (embedding_size) + LayerNorm + skip │
│         ↓                            │
│  Projector → Output (softplus)       │
└──────────────────────────────────────┘
```

Features:
- **Encoder**: Maps inputs to embedding space
- **Projector**: Maps embeddings to target predictions
- **Softplus output**: Ensures non-negative predictions

### Training Algorithm

```python
for epoch in range(epochs):
    for task in range(num_tasks):
        # Split data into support and query sets
        support_X, support_y = sample(k=num_shot)
        query_X, query_y = sample(k=num_query)
        
        # Create prototype from support set
        support_embeddings = encoder(support_X)
        prototype = mean(support_embeddings)
        prototype_target = mean(support_y)
        
        # Predict query targets using distance to prototype
        query_embeddings = encoder(query_X)
        distances = cdist(query_embeddings, prototype)
        weights = softmax(-distances)
        predicted = weights @ prototype_target
        
        loss = MSE(predicted, query_y)
```

---

## Comparison: Reptile vs ProtoNet

| Aspect | Reptile | ProtoNet |
|--------|---------|----------|
| **Learns** | Good initialization | Embedding space |
| **Adaptation** | Gradient-based | Distance-based |
| **Multi-target** | Separate models | Single model |
| **Min samples** | 3 | 10 |
| **Computation** | More inner steps | Faster inference |

---

## When to Use Meta-Learning Models

| Scenario | Recommended Model |
|----------|-------------------|
| **New material system with few samples** | ProtoNet or Reptile |
| **Transferring knowledge across projects** | Reptile |
| **Fast adaptation needed** | ProtoNet |
| **Complex feature interactions** | Reptile (skip connections) |
| **Stable, well-established system** | Standard RF/PINN |

---

## Hyperparameters

### Reptile

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 128 | Neurons per layer |
| `num_layers` | 3 | Hidden layers |
| `dropout_rate` | 0.3 | MC Dropout probability |
| `epochs` | Meta-epochs | Outer loop iterations |
| `inner_steps` | 10 | Gradient steps per task |
| `reptile_step` | 0.1→0.01 | Decaying step size |

### ProtoNet

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_size` | 256 | Embedding dimension |
| `num_layers` | 3 | Encoder layers |
| `dropout_rate` | 0.3 | Regularization |
| `num_shot` | 5 | Support set size |
| `num_query` | 5 | Query set size |
| `num_tasks` | 5 | Tasks per epoch |

---

## Technical Details

### Reptile File: `app/models/reptile_model.py`

**Key Classes**:
- `ReptileModel`: Single-target neural network
- `ReptileMultiTargetWrapper`: Manages multiple per-target models

**Key Functions**:
- `reptile_train()`: Meta-learning training loop
- `evaluate_reptile()`: WEBSLAMD utility scoring

### ProtoNet File: `app/models/protonet_model.py`

**Key Classes**:
- `ProtoNetModel`: Encoder + projector architecture

**Key Functions**:
- `protonet_train()`: Episode-based training
- `evaluate_protonet()`: WEBSLAMD utility scoring

---

## Practical Tips

1. **Use meta-learning when starting new projects** - they adapt quickly
2. **Reptile is more stable** with very small datasets (<10 samples)
3. **ProtoNet is faster at inference** due to prototype-based prediction
4. **Both use MC Dropout** for uncertainty - same as PINN
5. **Monitor uncertainty spread** - if all equal, exploration is ineffective
