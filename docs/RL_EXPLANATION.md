# Reinforcement Learning for Materials Discovery

## What is Reinforcement Learning (RL)?

Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning (which learns from labeled examples), RL learns from **experience** through trial and error.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Learning Loop                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌─────────┐    Action     ┌─────────────┐                   │
│    │  Agent  │──────────────▶│ Environment │                   │
│    │  (PPO)  │               │  (Design    │                   │
│    │         │◀──────────────│   Space)    │                   │
│    └─────────┘  State+Reward └─────────────┘                   │
│                                                                 │
│  Agent learns: "Which samples lead to the best outcomes?"       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **State**: Current knowledge (training data statistics, candidate features)
- **Action**: Select which sample to test next
- **Reward**: Feedback from testing (did prediction improve? did we find better materials?)
- **Policy**: Learned strategy for making selections

---

## Our Implementation: PPO (Proximal Policy Optimization)

We use **PPO** because it's:
- ✅ Stable with limited data
- ✅ Good for continuous feature spaces
- ✅ Balances exploration vs exploitation
- ✅ Works well in iterative improvement scenarios

### Architecture

```python
# PolicyNetwork (Actor-Critic)
┌──────────────────────────────────────┐
│  Input: State (candidate features)   │
│         ↓                            │
│  Shared Layers (128 neurons)         │
│         ↓                            │
│  ┌──────────┐    ┌──────────┐       │
│  │  Actor   │    │  Critic  │       │
│  │ (scores) │    │ (value)  │       │
│  └──────────┘    └──────────┘       │
│         ↓              ↓            │
│  Selection Score   Value Estimate   │
└──────────────────────────────────────┘
```

### State Representation

For each candidate sample, the state includes:
1. **Candidate features** (normalized)
2. **Training data statistics** (mean of labeled samples)
3. **Novelty indicator** (distance from training data centroid)

This tells the agent: "Given what I've learned so far, how promising is this sample?"

---

## How RL Discovers the Design Space

### Phase 1: Exploration (Early Cycles)

When the RL agent is new, it has limited knowledge. It combines:
- **30% RL Score**: Initial learned preferences
- **70% Standard Acquisition**: Proven methods (UCB/EI)

This ensures safe, reasonable selections while the agent learns.

### Phase 2: Learning (Ongoing)

After each cycle, the agent receives **dual rewards**:

```
Reward = 0.5 × (RMSE Improvement) + 0.5 × (Target Discovery)
```

| Reward Component | What It Measures |
|------------------|------------------|
| RMSE Improvement | Did prediction accuracy get better? |
| Target Discovery | Did we find materials with better properties? |

The agent learns patterns like:
- "Samples far from training data often improve the model"
- "Certain feature combinations lead to high-performing materials"
- "Balancing exploration and exploitation gives best results"

### Phase 3: Exploitation (Mature Agent)

As the agent accumulates experience, it learns:
- Which regions of the design space are most promising
- What sample characteristics correlate with good outcomes
- When to explore new areas vs exploit known good regions

---

## Why RL is Good for Materials Discovery

### 1. **Adaptive Strategy**

Traditional methods (GP, Random Forest) use fixed acquisition functions. RL **adapts** to your specific:
- Dataset characteristics
- Optimization objectives
- Experimental constraints

### 2. **Multi-Objective Optimization**

Materials discovery often has competing goals:
- High strength + Low cost
- High durability + Low CO2 emissions

RL naturally handles these trade-offs through its reward structure.

### 3. **Learning from History**

RL remembers what worked:
- "Last time I selected samples with high silica content, we found strong materials"
- "Exploring the edge of the design space improved predictions"

This institutional memory compounds over cycles.

### 4. **Efficiency with Limited Data**

In materials science, each experiment is expensive. RL:
- Learns from every selection (good or bad)
- Quickly identifies promising regions
- Avoids wasting resources on poor candidates

### 5. **Transferable Knowledge**

Experience from one material system can inform another:
- Optimization strategies transfer across projects
- The agent becomes more efficient over time

---

## Comparison with Other Models

| Model | Strengths | When RL is Better |
|-------|-----------|-------------------|
| **GP/DKL** | Excellent uncertainty | When you need adaptive strategy |
| **Random Forest** | Fast, robust | When you want to learn from patterns |
| **PINN** | Physics constraints | When domain knowledge is incomplete |
| **RL (PPO)** | Learns from experience | For iterative, long-term campaigns |

---

## Practical Tips

1. **Start with Hybrid Mode** (RL + LLM) for best results
2. **Run multiple cycles** - RL improves with experience
3. **Sync lab results** - feedback is essential for learning
4. **Be patient** - RL's value compounds over time

---

## Technical Details

**File**: `app/models/rl_model.py`

**Key Classes**:
- `PolicyNetwork`: Neural network for decision-making
- `ExperienceBuffer`: Stores learning experiences
- `RLModel`: Main model class with train/predict methods

**Key Functions**:
- `train_rl_model()`: Trains the RL agent
- `evaluate_rl_model()`: Scores candidates using learned policy

**Hyperparameters**:
- Hidden dimension: 128
- Learning rate: 0.001
- PPO clip epsilon: 0.2
- RL weight in final utility: 0.3 (30%)
