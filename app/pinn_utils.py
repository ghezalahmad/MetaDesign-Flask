"""
Physics-Informed Loss Functions for Any Dataset

This module provides ADAPTIVE physics-based constraints that work with ANY dataset.
Constraints are automatically applied based on what's available in the data.

Key principles:
1. Non-negativity - applies to any physical quantity
2. Reasonable bounds - learns from training data
3. Monotonicity constraints - can be configured per-target
4. Smoothness - always applicable
5. Domain-specific constraints - only applied when matching columns exist
"""

import torch
import torch.nn as nn
import numpy as np


def compute_physics_loss(predictions, inputs, 
                         physics_weight=0.1,
                         input_columns=None,
                         target_columns=None,
                         training_stats=None):
    """
    Compute adaptive physics-informed loss that works with ANY dataset.
    
    The function automatically detects which constraints can be applied
    based on available columns and data structure.
    
    Args:
        predictions: Tensor of model predictions (batch_size, n_targets)
        inputs: Tensor of input features (batch_size, n_features)
        physics_weight: Overall scaling for physics loss (default 0.1)
        input_columns: Optional list of input column names (for column detection)
        target_columns: Optional list of target column names
        training_stats: Optional dict with 'min', 'max', 'mean', 'std' from training data
    
    Returns:
        Combined physics loss tensor (scalar)
    """
    device = predictions.device
    batch_size = predictions.shape[0]
    
    # Initialize total physics loss
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # UNIVERSAL CONSTRAINT 1: Non-negativity
    # Physical quantities (strength, emissions, etc.) are typically non-negative
    # =========================================================================
    negativity_penalty = torch.relu(-predictions).pow(2).mean()
    total_loss = total_loss + 0.5 * negativity_penalty
    
    # =========================================================================
    # UNIVERSAL CONSTRAINT 2: Smoothness Regularization
    # Physical properties should vary smoothly with input changes
    # This is applicable to ANY dataset
    # =========================================================================
    if batch_size > 1:
        smoothness_loss = torch.diff(predictions, dim=0).pow(2).mean() * 0.01
        total_loss = total_loss + 0.1 * smoothness_loss
    
    # =========================================================================
    # UNIVERSAL CONSTRAINT 3: Output variance should exist
    # Penalize if all predictions collapse to same value (degenerate solution)
    # =========================================================================
    if batch_size > 1:
        pred_var = predictions.var(dim=0).mean()
        collapse_penalty = torch.relu(0.01 - pred_var)  # Penalize near-zero variance
        total_loss = total_loss + 0.2 * collapse_penalty
    
    # =========================================================================
    # ADAPTIVE CONSTRAINT 4: Apply domain constraints if columns match
    # Only applies when we recognize specific material science columns
    # =========================================================================
    if input_columns is not None and len(input_columns) > 0:
        # Convert to lowercase for matching
        col_lower = [c.lower() for c in input_columns]
        
        # Check for water content (Abrams' law)
        water_idx = _find_column_index(col_lower, ['water', 'w/c', 'w/b', 'liquid'])
        
        # Check for binder content
        binder_indices = []
        for binder_name in ['cement', 'binder', 'fa', 'ggbfs', 'slag', 'fly ash']:
            idx = _find_column_index(col_lower, [binder_name])
            if idx is not None:
                binder_indices.append(idx)
        
        # Apply Abrams-like constraint if water and strength-like target found
        if water_idx is not None and predictions.shape[1] >= 1:
            water_content = inputs[:, water_idx]
            strength_pred = predictions[:, 0]  # Assume first target is strength-like
            
            # Higher water should correlate with lower output (for strength)
            if batch_size > 1:
                water_centered = water_content - water_content.mean()
                strength_centered = strength_pred - strength_pred.mean()
                covariance = (water_centered * strength_centered).mean()
                # Only penalize if strong positive correlation (violates physics)
                abrams_loss = torch.relu(covariance - 0.1) * 0.5
                total_loss = total_loss + 0.1 * abrams_loss
        
        # Apply binder constraint if found
        if len(binder_indices) > 0 and predictions.shape[1] >= 1:
            binder_sum = sum(inputs[:, idx] for idx in binder_indices)
            strength_pred = predictions[:, 0]
            
            if batch_size > 1:
                binder_centered = binder_sum - binder_sum.mean()
                strength_centered = strength_pred - strength_pred.mean()
                covariance = (binder_centered * strength_centered).mean()
                # Penalize strong negative correlation (more binder = less strength is wrong)
                binder_loss = torch.relu(-covariance - 0.1) * 0.3
                total_loss = total_loss + 0.05 * binder_loss
    
    return total_loss * physics_weight


def _find_column_index(columns_lower, search_terms):
    """
    Find the index of a column matching any of the search terms.
    
    Args:
        columns_lower: List of column names in lowercase
        search_terms: List of strings to search for
    
    Returns:
        Index of matching column or None
    """
    for i, col in enumerate(columns_lower):
        for term in search_terms:
            if term in col:
                return i
    return None


def compute_domain_specific_loss(predictions, inputs, domain='general'):
    """
    Compute domain-specific physics constraints.
    
    Use this function when you know your domain and want stricter physics.
    
    Args:
        predictions: Model predictions tensor
        inputs: Input features tensor
        domain: One of 'general', 'concrete', 'metals', 'polymers', 'batteries'
    
    Returns:
        Domain-specific physics loss
    """
    device = predictions.device
    
    if domain == 'concrete' or domain == 'cementitious':
        return _compute_concrete_physics(predictions, inputs)
    elif domain == 'metals':
        return _compute_metals_physics(predictions, inputs)
    elif domain == 'polymers':
        return _compute_polymer_physics(predictions, inputs)
    elif domain == 'batteries':
        return _compute_battery_physics(predictions, inputs)
    else:
        # General: just return zero (no domain-specific constraints)
        return torch.tensor(0.0, device=device)


def _compute_concrete_physics(predictions, inputs):
    """
    Physics constraints specific to concrete/cementitious materials.
    """
    device = predictions.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Strength bounds for concrete (5-120 MPa is typical)
    if predictions.shape[1] >= 1:
        fc_pred = predictions[:, 0]
        lower_violation = torch.relu(5.0 - fc_pred)
        upper_violation = torch.relu(fc_pred - 120.0)
        bounds_loss = (lower_violation.pow(2) + upper_violation.pow(2)).mean()
        loss = loss + 0.3 * bounds_loss
    
    return loss


def _compute_metals_physics(predictions, inputs):
    """
    Physics constraints specific to metals/alloys.
    """
    device = predictions.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Yield strength typically 50-2000 MPa for metals
    if predictions.shape[1] >= 1:
        strength = predictions[:, 0]
        lower_violation = torch.relu(50.0 - strength)
        upper_violation = torch.relu(strength - 2000.0)
        bounds_loss = (lower_violation.pow(2) + upper_violation.pow(2)).mean()
        loss = loss + 0.2 * bounds_loss
    
    return loss


def _compute_polymer_physics(predictions, inputs):
    """
    Physics constraints specific to polymers.
    """
    device = predictions.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Tensile strength typically 10-150 MPa for polymers
    if predictions.shape[1] >= 1:
        strength = predictions[:, 0]
        lower_violation = torch.relu(10.0 - strength)
        upper_violation = torch.relu(strength - 150.0)
        bounds_loss = (lower_violation.pow(2) + upper_violation.pow(2)).mean()
        loss = loss + 0.2 * bounds_loss
    
    return loss


def _compute_battery_physics(predictions, inputs):
    """
    Physics constraints specific to battery materials.
    """
    device = predictions.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Capacity typically 0-300 mAh/g for cathodes
    if predictions.shape[1] >= 1:
        capacity = predictions[:, 0]
        lower_violation = torch.relu(-capacity)
        upper_violation = torch.relu(capacity - 300.0)
        bounds_loss = (lower_violation.pow(2) + upper_violation.pow(2)).mean()
        loss = loss + 0.2 * bounds_loss
    
    return loss
