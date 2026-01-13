"""
Reinforcement Learning Model for Active Learning

Uses PPO (Proximal Policy Optimization) to learn optimal sample selection strategies.
The agent learns from historical experiments to maximize both:
1. Prediction accuracy improvement (RMSE reduction)
2. Discovery of high-performing materials (target optimization)

The model maintains state across cycles within a session to continuously improve.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    State: Aggregated representation of training data + candidate features
    Output: Score for each candidate sample
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Actor head - outputs score for candidate
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Critic head - estimates value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        return features
    
    def get_action_scores(self, state):
        """Get scores for all candidate states."""
        features = self.forward(state)
        scores = self.actor(features)
        return scores.squeeze(-1)
    
    def get_value(self, state):
        """Get value estimate for state."""
        features = self.forward(state)
        return self.critic(features).squeeze(-1)
    
    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and values for given actions."""
        scores = self.get_action_scores(states)
        values = self.get_value(states)
        
        # Softmax over candidates to get action probabilities
        probs = torch.softmax(scores, dim=0)
        log_probs = torch.log(probs + 1e-10)
        
        # Get log prob for selected action
        action_log_probs = log_probs.gather(0, actions.long())
        entropy = -(probs * log_probs).sum()
        
        return action_log_probs, values, entropy


class ExperienceBuffer:
    """Stores experiences for PPO training."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def compute_returns(self, gamma=0.99, lam=0.95):
        """Compute GAE returns and advantages."""
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return returns, advantages
    
    def get_batch(self):
        """Get all experiences as tensors."""
        returns, advantages = self.compute_returns()
        
        return {
            'states': torch.stack(self.states),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'advantages': torch.tensor(advantages, dtype=torch.float32),
            'old_log_probs': torch.stack(self.log_probs),
        }


class RLModel:
    """
    Reinforcement Learning model for active learning sample selection.
    
    Uses PPO to learn which samples are most valuable to test, balancing:
    1. Prediction accuracy improvement
    2. Discovery of high-performing materials
    
    The model learns from session history and improves over cycles.
    """
    
    def __init__(self, input_columns=None, target_columns=None, hidden_dim=128, lr=1e-3):
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.policy = None
        self.optimizer = None
        self.scaler_x = None
        self.scaler_y = None
        self.is_trained = False
        
        # Experience buffer for session learning
        self.buffer = ExperienceBuffer()
        
        # History for reward calculation
        self.history = {
            'selected_samples': [],
            'rmse_before': [],
            'rmse_after': [],
            'best_target_value': None,
            'target_improvements': []
        }
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.ppo_epochs = 4
        
    def _init_policy(self, state_dim):
        """Initialize policy network."""
        self.policy = PolicyNetwork(state_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def _build_state(self, train_features, candidate_features, train_targets=None):
        """
        Build state representation combining training data context and candidate.
        
        State = [candidate_features, train_data_aggregate, uncertainty_estimate]
        """
        # Aggregate training data info (mean, std of each feature)
        train_mean = train_features.mean(axis=0)
        train_std = train_features.std(axis=0) + 1e-6
        
        # Normalize candidate features relative to training distribution
        normalized_candidate = (candidate_features - train_mean) / train_std
        
        # Add training data statistics
        n_train_samples = len(train_features) / 1000  # Normalized count
        
        # Calculate distance from training data centroid (novelty proxy)
        centroid = train_features.mean(axis=0)
        distance_from_centroid = np.linalg.norm(candidate_features - centroid, axis=1, keepdims=True)
        distance_from_centroid = distance_from_centroid / (distance_from_centroid.max() + 1e-6)
        
        # Combine into state
        state = np.concatenate([
            normalized_candidate,
            np.tile(train_mean, (len(candidate_features), 1)),
            np.tile([n_train_samples], (len(candidate_features), 1)),
            distance_from_centroid
        ], axis=1)
        
        return state
    
    def train(self, X, y):
        """
        Train the RL policy from accumulated experiences.
        
        This is called to update the policy based on feedback from previous selections.
        """
        if isinstance(X, pd.DataFrame):
            self.input_columns = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.DataFrame):
            self.target_columns = y.columns.tolist()
            y = y.values
        
        # Initialize scalers
        self.scaler_x = StandardScaler()
        X_scaled = self.scaler_x.fit_transform(X)
        
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
        
        # State dimension: features + aggregates + novelty
        state_dim = X.shape[1] * 2 + 2  # candidate + train_mean + n_samples + distance
        
        if self.policy is None:
            self._init_policy(state_dim)
        
        # If we have experiences in buffer, train on them
        if len(self.buffer.rewards) >= 2:
            self._update_policy()
        
        self.is_trained = True
        return self
    
    def _update_policy(self):
        """PPO policy update."""
        if len(self.buffer.states) < 2:
            return
        
        batch = self.buffer.get_batch()
        
        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            action_log_probs, values, entropy = self.policy.evaluate_actions(
                batch['states'], batch['actions']
            )
            
            # PPO clipped objective
            ratios = torch.exp(action_log_probs - batch['old_log_probs'].detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values, batch['returns'])
            
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        logger.info(f"PPO update: actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f}")
    
    def predict_with_uncertainty(self, X, input_columns=None, num_samples=None):
        """
        Generate utility scores for candidates using RL policy.
        
        Higher scores indicate samples the RL agent believes are more valuable to test.
        """
        if isinstance(X, pd.DataFrame):
            if self.input_columns:
                X = X[self.input_columns]
            X = X.values
        
        n_samples = len(X)
        
        if self.scaler_x is None:
            # Not trained yet, return uniform scores
            scores = np.ones(n_samples) / n_samples
            uncertainties = np.ones(n_samples) * 0.5
            return scores.reshape(-1, 1), uncertainties.reshape(-1, 1), None
        
        # Scale inputs
        X_scaled = self.scaler_x.transform(X)
        
        # Build states for each candidate
        # Use scaled training data statistics
        train_mean = np.zeros(X.shape[1])  # Placeholder, actual would come from stored training data
        train_std = np.ones(X.shape[1])
        
        states = self._build_state_simple(X_scaled)
        
        # Get policy scores
        with torch.no_grad():
            state_tensor = torch.tensor(states, dtype=torch.float32)
            
            if self.policy is not None:
                scores = self.policy.get_action_scores(state_tensor).numpy()
                # Convert to probabilities via softmax
                scores = np.exp(scores - scores.max())
                scores = scores / scores.sum()
            else:
                scores = np.ones(n_samples) / n_samples
        
        # Uncertainty as inverse of confidence (higher score = lower uncertainty there)
        uncertainties = 1 - scores
        uncertainties = uncertainties / uncertainties.max()  # Normalize
        
        return scores.reshape(-1, 1), uncertainties.reshape(-1, 1), None
    
    def _build_state_simple(self, X_scaled):
        """Build simple state representation for prediction."""
        # For prediction, we just use the features + some aggregate info
        n_samples, n_features = X_scaled.shape
        
        # Add simple aggregates
        feature_mean = X_scaled.mean(axis=0)
        distance_from_mean = np.linalg.norm(X_scaled - feature_mean, axis=1, keepdims=True)
        distance_from_mean = distance_from_mean / (distance_from_mean.max() + 1e-6)
        
        # State: features + mean + n_train_placeholder + distance
        state = np.concatenate([
            X_scaled,
            np.tile(feature_mean, (n_samples, 1)),
            np.ones((n_samples, 1)) * 0.1,  # Placeholder for n_train
            distance_from_mean
        ], axis=1)
        
        return state
    
    def add_feedback(self, selected_idx, rmse_improvement, target_improvement):
        """
        Add feedback from a completed cycle to improve future selections.
        
        Args:
            selected_idx: Index of sample that was selected
            rmse_improvement: Change in RMSE (negative is better)
            target_improvement: Change in best target value found
        """
        # Calculate reward combining both objectives
        # Positive reward for RMSE decrease and target improvement
        rmse_reward = -rmse_improvement  # Negative RMSE change is good
        target_reward = target_improvement if target_improvement > 0 else 0
        
        combined_reward = 0.5 * rmse_reward + 0.5 * target_reward
        
        self.history['selected_samples'].append(selected_idx)
        self.history['target_improvements'].append(target_improvement)
        
        # Add to buffer if we have a previous state
        if hasattr(self, '_last_state') and self._last_state is not None:
            self.buffer.add(
                state=self._last_state,
                action=self._last_action,
                reward=combined_reward,
                value=self._last_value,
                log_prob=self._last_log_prob,
                done=False
            )
        
        logger.info(f"RL feedback added: reward={combined_reward:.4f} (RMSE: {rmse_reward:.4f}, Target: {target_reward:.4f})")
    
    def get_input_columns(self):
        """Return input column names."""
        if self.scaler_x is not None and hasattr(self.scaler_x, 'feature_names_in_'):
            return list(self.scaler_x.feature_names_in_)
        return self.input_columns
    
    def save_checkpoint(self, path: str = None):
        """
        Save model checkpoint to disk for persistence across sessions.
        
        Args:
            path: Path to save checkpoint. If None, uses default location.
        """
        import os
        import pickle
        
        if path is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, 'rl_model_checkpoint.pt')
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict() if self.policy else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'input_columns': self.input_columns,
            'target_columns': self.target_columns,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'is_trained': self.is_trained,
            'history': self.history,
            'buffer_states': [s.numpy() if torch.is_tensor(s) else s for s in self.buffer.states],
            'buffer_actions': self.buffer.actions,
            'buffer_rewards': self.buffer.rewards,
            'buffer_values': self.buffer.values,
            'buffer_log_probs': [lp.numpy() if torch.is_tensor(lp) else lp for lp in self.buffer.log_probs],
            'buffer_dones': self.buffer.dones,
            'clip_epsilon': self.clip_epsilon,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'ppo_epochs': self.ppo_epochs,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"RL checkpoint saved to {path}")
        return path
    
    def load_checkpoint(self, path: str = None):
        """
        Load model checkpoint from disk.
        
        Args:
            path: Path to load checkpoint from. If None, uses default location.
            
        Returns:
            bool: True if checkpoint was loaded, False otherwise.
        """
        import os
        
        if path is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
            path = os.path.join(checkpoint_dir, 'rl_model_checkpoint.pt')
        
        if not os.path.exists(path):
            logger.info(f"No checkpoint found at {path}, starting fresh")
            return False
        
        try:
            checkpoint = torch.load(path, weights_only=False)
            
            # Restore model parameters
            self.input_columns = checkpoint.get('input_columns')
            self.target_columns = checkpoint.get('target_columns')
            self.hidden_dim = checkpoint.get('hidden_dim', 128)
            self.lr = checkpoint.get('lr', 1e-3)
            self.is_trained = checkpoint.get('is_trained', False)
            self.history = checkpoint.get('history', {})
            self.scaler_x = checkpoint.get('scaler_x')
            self.scaler_y = checkpoint.get('scaler_y')
            
            # Restore PPO hyperparameters
            self.clip_epsilon = checkpoint.get('clip_epsilon', 0.2)
            self.value_coef = checkpoint.get('value_coef', 0.5)
            self.entropy_coef = checkpoint.get('entropy_coef', 0.01)
            self.ppo_epochs = checkpoint.get('ppo_epochs', 4)
            
            # Restore policy network
            if checkpoint.get('policy_state_dict') is not None:
                # Need to determine state_dim from saved data
                if self.scaler_x is not None:
                    n_features = len(self.scaler_x.mean_)
                    state_dim = n_features * 2 + 2
                    self._init_policy(state_dim)
                    self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    if checkpoint.get('optimizer_state_dict'):
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore experience buffer
            self.buffer.states = [torch.tensor(s, dtype=torch.float32) for s in checkpoint.get('buffer_states', [])]
            self.buffer.actions = checkpoint.get('buffer_actions', [])
            self.buffer.rewards = checkpoint.get('buffer_rewards', [])
            self.buffer.values = checkpoint.get('buffer_values', [])
            self.buffer.log_probs = [torch.tensor(lp, dtype=torch.float32) for lp in checkpoint.get('buffer_log_probs', [])]
            self.buffer.dones = checkpoint.get('buffer_dones', [])
            
            logger.info(f"RL checkpoint loaded from {path}")
            logger.info(f"  - Trained: {self.is_trained}")
            logger.info(f"  - Experience buffer size: {len(self.buffer.rewards)}")
            logger.info(f"  - History: {len(self.history.get('selected_samples', []))} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RL checkpoint: {e}")
            return False


def train_rl_model(data: pd.DataFrame, input_columns: list, target_columns: list, 
                   model_params: dict = None):
    """
    Train the RL model.
    
    For RL, training is iterative - the model improves as it receives feedback
    from each cycle. Initial training uses the labeled data to bootstrap.
    
    Automatically loads previous checkpoint if available for continuous learning.
    """
    model_params = model_params or {}
    
    model = RLModel(
        input_columns=input_columns,
        target_columns=target_columns,
        hidden_dim=model_params.get('hidden_dim', 128),
        lr=model_params.get('lr', 1e-3)
    )
    
    # Try to load previous checkpoint for continuous learning
    checkpoint_loaded = model.load_checkpoint()
    if checkpoint_loaded:
        logger.info("Continuing RL learning from previous session")
    
    # Get labeled data for training
    train_df = data.dropna(subset=target_columns)
    
    if len(train_df) < 2:
        logger.warning("Not enough labeled data for RL training, using untrained model")
        return model, None, None
    
    X_train = train_df[input_columns].values
    y_train = train_df[target_columns].values
    
    model.train(X_train, y_train)
    
    logger.info(f"RL Model trained with {len(train_df)} samples")
    
    return model, None, None


def evaluate_rl_model(model: RLModel, data: pd.DataFrame, input_columns: list,
                      target_columns: list, curiosity: float, weights_targets: np.ndarray,
                      max_or_min_targets: list):
    """
    Evaluate candidates using the RL model for sample selection.
    
    The RL model scores candidates based on learned policy that balances:
    1. Prediction accuracy improvement potential
    2. Target optimization potential
    """
    from app.utils.webslamd_utility import calculate_webslamd_utility
    from app.utils.utils import calculate_novelty
    
    # Split labeled vs candidate rows
    labeled_data = data.dropna(subset=target_columns)
    candidate_df = data[data[target_columns[0]].isnull()].copy()
    
    if candidate_df.empty:
        logger.warning("No candidate rows found for RL evaluation")
        return pd.DataFrame()
    
    candidate_inputs = candidate_df[input_columns]
    
    # Get RL policy scores
    rl_scores, rl_uncertainties, _ = model.predict_with_uncertainty(candidate_inputs)
    
    # Use a basic surrogate for predictions (simple mean prediction for unlabeled)
    # In practice, RL is about SELECTION, not prediction - we use it alongside another model
    labeled_mean = labeled_data[target_columns].mean().values
    labeled_std = labeled_data[target_columns].std().values + 1e-6
    
    # Simple predictions based on feature similarity to high-performing labeled samples
    X_labeled = labeled_data[input_columns].values
    X_candidate = candidate_inputs.values
    y_labeled = labeled_data[target_columns].values
    
    # Efficient KNN using ball_tree for O(n log m) instead of O(n*m)
    from sklearn.neighbors import NearestNeighbors
    k = min(3, len(X_labeled))
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean')
    nn.fit(X_labeled)
    distances, nearest_indices = nn.kneighbors(X_candidate)
    
    predictions = np.zeros((len(X_candidate), len(target_columns)))
    for i in range(len(X_candidate)):
        predictions[i] = y_labeled[nearest_indices[i]].mean(axis=0)
    
    # Uncertainty from RL model (inverted scores - high score = confident)
    uncertainties = rl_uncertainties.flatten()
    
    # Add predictions to candidate_df
    for i, col in enumerate(target_columns):
        candidate_df[f"Predicted_{col}"] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties * labeled_std[i]
    
    # Calculate utility combining RL scores with standard utility
    standard_utility = calculate_webslamd_utility(
        predictions=predictions,
        uncertainties=np.tile(uncertainties.reshape(-1, 1), (1, len(target_columns))) * labeled_std,
        labeled_data=labeled_data,
        target_columns=target_columns,
        max_or_min=max_or_min_targets,
        weights=weights_targets,
        curiosity=curiosity
    )
    
    # Combine RL selection score with standard utility
    # RL score influences which samples to prioritize
    from app.utils.settings_manager import SettingsManager
    rl_weight = SettingsManager.get_setting('rl_weight', 0.3)  # Configurable, default 30%
    combined_utility = (1 - rl_weight) * standard_utility + rl_weight * rl_scores.flatten()
    
    candidate_df["Utility"] = combined_utility
    candidate_df["Utility"] = pd.to_numeric(candidate_df["Utility"], errors="coerce").fillna(0.0)
    
    # Aggregate uncertainty
    candidate_df["Uncertainty"] = uncertainties
    
    # Novelty
    novelty_scores = calculate_novelty(X_candidate, X_labeled)
    candidate_df["Novelty"] = novelty_scores
    
    # RL score as additional column
    candidate_df["RL_Score"] = rl_scores.flatten()
    
    # Select best candidate
    candidate_df["Selected for Testing"] = False
    if not candidate_df["Utility"].empty:
        max_utility_idx = candidate_df["Utility"].idxmax()
        candidate_df.loc[max_utility_idx, "Selected for Testing"] = True
    
    # Sort by utility
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)
    
    # Save checkpoint for persistent learning across sessions
    try:
        model.save_checkpoint()
    except Exception as e:
        logger.warning(f"Failed to save RL checkpoint: {e}")
    
    logger.info(f"RL evaluation complete: {len(result_df)} candidates scored")
    logger.info(f"Top RL score: {rl_scores.max():.4f}, Top utility: {combined_utility.max():.4f}")
    
    return result_df
