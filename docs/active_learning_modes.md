# Active Learning Modes: Detailed Documentation

This document explains the three active learning modes in MetaDesign: **ML Mode**, **LLM Agent Mode**, and **Hybrid Mode**. Each mode represents a different approach to navigating the design space and selecting the next experiments to run.

---

## Overview: The Three Modes

| Mode | Engine | Exploration | Exploitation | Predictions |
|------|--------|-------------|--------------|-------------|
| **ML Mode** | Surrogate Model + Acquisition Function | Uncertainty-driven | Prediction-driven | ML predictions with uncertainty |
| **LLM Agent Mode** | LLM Reasoning | Domain knowledge | Pattern recognition | LLM-estimated values |
| **Hybrid Mode** | ML + LLM Fusion | Both mechanisms | Both mechanisms | ML predictions + semantic guidance |

---

## 1. ML Mode (Machine Learning Only)

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML MODE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Train Surrogate Model on labeled data                      │
│  2. Predict target values for all unlabeled candidates         │
│  3. Estimate uncertainty for each prediction                   │
│  4. Calculate Utility = f(prediction, uncertainty, curiosity)  │
│  5. Select top-N samples with highest utility                  │
└─────────────────────────────────────────────────────────────────┘
```

### Exploration vs Exploitation

The **curiosity parameter** (0.0 to 1.0) controls the balance:

| Curiosity | Behavior | Best For |
|-----------|----------|----------|
| 0.0 | Pure exploitation - select highest predicted values | Fine-tuning known good regions |
| 0.5 | Balanced - consider both prediction and uncertainty | General optimization |
| 1.0 | Pure exploration - select most uncertain samples | Early-stage discovery |

### Acquisition Functions

- **WEBSLAMD (Default)**: Normalized predictions + weighted uncertainty
- **UCB (Upper Confidence Bound)**: Prediction + β × uncertainty
- **EI (Expected Improvement)**: Probability of improving over current best
- **Thompson Sampling**: Stochastic sampling from posterior distribution

---

## 2. LLM Agent Mode (Pure LLM Reasoning)

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM AGENT MODE PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Prepare experimental history (few-shot examples)           │
│  2. Construct prompt with:                                      │
│     - Optimization objectives (maximize/minimize)              │
│     - Parameter space description                               │
│     - Historical experiments (context)                          │
│     - Strategy instructions (explore/exploit/balanced)          │
│  3. LLM proposes optimal parameters                             │
│  4. Match proposal to nearest candidate in design space         │
│  5. LLM predicts expected target values for selected sample    │
└─────────────────────────────────────────────────────────────────┘
```

### LLM Strategy Settings

| Strategy | LLM Prompt Guidance | Behavior |
|----------|---------------------|----------|
| **Explore** | "Choose values VERY DIFFERENT from all previous experiments" | Maximizes diversity in parameter space |
| **Exploit** | "Focus on parameters similar to BEST performing experiments" | Refines known promising regions |
| **Balanced** | "Balance exploration and exploitation based on your judgment" | LLM decides based on iteration |

### Key Characteristics

- **No model training** - Relies purely on LLM's domain knowledge
- **Zero-shot capable** - Can work even with minimal labeled data
- **Domain reasoning** - LLM uses materials science knowledge to infer relationships
- **Prediction via reasoning** - LLM estimates target values based on patterns

### Limitations

- Predictions are estimates, not statistically grounded
- No uncertainty quantification
- Performance depends on LLM's domain knowledge

---

## 3. Hybrid Mode (ML + LLM Fusion) ⭐ Most Sophisticated

### How It Works

The Hybrid mode combines the **statistical rigor of ML** with the **domain reasoning of LLMs** to create a more robust optimization strategy.

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID MODE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: ML Surrogate (Data-Driven)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Train surrogate model on labeled data                │   │
│  │  • Predict values for all candidates                    │   │
│  │  • Calculate ML Utility (prediction + uncertainty)      │   │
│  │  • Output: ML_Utility for each candidate               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  STEP 2: LLM Proposal (Knowledge-Driven)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Send experimental history to LLM                     │   │
│  │  • LLM generates ideal experiment parameters            │   │
│  │  • Output: Proposed parameter values as text            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  STEP 3: Semantic Matching                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Convert each candidate to text description           │   │
│  │  • TF-IDF vectorization of LLM proposal + candidates    │   │
│  │  • Cosine similarity = Semantic_Score                   │   │
│  │  • Output: Semantic score for each candidate            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  STEP 4: Score Fusion                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Final_Utility = (w_ml × ML_Utility) +                  │   │
│  │                  (w_llm × Semantic_Score)               │   │
│  │                                                          │   │
│  │  Default weights: w_ml = 0.5, w_llm = 0.5               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  STEP 5: Batch Selection                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Select top-N samples by fused utility                │   │
│  │  • Apply diversity-aware selection (30% weight)         │   │
│  │  • Mark as "Selected for Testing"                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Fusion Formula

```python
Final_Utility = (w_ml × normalized_ML_Utility) + (w_llm × Semantic_Score)
```

Where:
- **ML_Utility** (normalized 0-1): From surrogate model predictions + uncertainty
- **Semantic_Score** (0-1): Cosine similarity between LLM proposal and candidate
- **w_ml** and **w_llm**: User-configurable weights (default: 0.5 each)

### Role of Each Component

#### ML Component (Statistical Exploitation)
| Aspect | Contribution |
|--------|--------------|
| **Data efficiency** | Quantifies relationships from experimental data |
| **Uncertainty estimation** | Identifies regions with limited data |
| **Exploitation** | High scores for predicted high-performers |
| **Exploration** | High scores for uncertain regions (via curiosity) |

#### LLM Component (Knowledge-Driven Guidance)
| Aspect | Contribution |
|--------|--------------|
| **Domain knowledge** | Injects materials science heuristics |
| **Pattern recognition** | Identifies trends beyond statistical patterns |
| **Semantic reasoning** | "Similar formulations should behave similarly" |
| **Constraint awareness** | Respects physical/chemical constraints |

### How Hybrid Balances Exploration vs Exploitation

```
High ML_Utility + High Semantic_Score
    → Strong candidate (both signals agree)
    
High ML_Utility + Low Semantic_Score  
    → Data suggests it's good, but LLM disagrees
    → May still be selected if w_ml > w_llm
    
Low ML_Utility + High Semantic_Score
    → LLM domain knowledge recommends it
    → Explores areas ML might miss due to limited data
    
Low ML_Utility + Low Semantic_Score
    → Neither signal recommends it → low priority
```

### Weight Tuning Guidelines

| w_ml : w_llm | Use Case |
|--------------|----------|
| **0.8 : 0.2** | Trust data - Large labeled dataset, well-characterized system |
| **0.5 : 0.5** | Balanced - Default, recommended for most cases |
| **0.3 : 0.7** | Trust LLM - Small dataset, LLM has strong domain knowledge |
| **0.0 : 1.0** | Pure LLM - Equivalent to LLM Agent mode |
| **1.0 : 0.0** | Pure ML - Equivalent to ML mode |

---

## Visualization: Understanding the t-SNE Plot

Each mode visualizes differently:

### ML Mode
- **Circles (color gradient)**: Predicted candidates, colored by Utility
- **Crosses**: Labeled training data

### LLM Agent Mode
- **Gray crosses**: Unlabeled candidates
- **Green circles**: Labeled experiments (training data)
- **Red star**: LLM-selected point for testing

### Hybrid Mode
Same as ML mode, but utility incorporates both ML and semantic scores.

---

## Summary: When to Use Each Mode

| Scenario | Recommended Mode |
|----------|------------------|
| Large labeled dataset, well-understood system | **ML Mode** |
| Very small dataset (< 10 labels), need domain reasoning | **LLM Agent Mode** |
| Moderate dataset, want best of both worlds | **Hybrid Mode** |
| Initial exploration, unknown parameter space | **LLM Agent (Explore strategy)** |
| Fine-tuning known promising region | **ML Mode (low curiosity)** |
| Production optimization with both data and expertise | **Hybrid Mode** |

---

## Technical Details

### Files Involved

| Component | File |
|-----------|------|
| Mode Router | `app/engines/hybrid_engine.py` |
| ML Engine | `app/engines/ml_engine.py` |
| LLM Agent | `app/engines/llm_agent.py` |
| Semantic Matcher | `app/engines/semantic_matcher.py` |
| Batch Selection | `app/utils/batch_selector.py` |
| Visualizations | `app/utils/plot_generator.py` |

### API Configuration

```json
{
  "active_learning_mode": "HYBRID_MODE",  // or "ML_MODE", "LLM_AGENT_MODE"
  "hybrid_weights": {"w_ml": 0.5, "w_llm": 0.5},
  "curiosity": 0.5,
  "batch_size": 3,
  "llm_strategy": "balanced"  // "explore", "exploit", "balanced"
}
```
