"""
Acquisition Function Package

Provides pluggable acquisition functions for active learning:
- WEBSLAMD: Default utility matching the WEBSLAMD framework
- UCB: Upper Confidence Bound
- EI: Expected Improvement
- Thompson: Thompson Sampling
"""

from app.acquisition.base import (
    AcquisitionFunction,
    WEBSLAMD,
    UCB,
    ExpectedImprovement,
    ThompsonSampling,
    get_acquisition_function,
    ACQUISITION_FUNCTIONS
)

__all__ = [
    'AcquisitionFunction',
    'WEBSLAMD',
    'UCB',
    'ExpectedImprovement',
    'ThompsonSampling',
    'get_acquisition_function',
    'ACQUISITION_FUNCTIONS'
]
