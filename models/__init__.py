"""
models package: architecture primitives for the multi-task MNIST geodesic experiment.

Exports:
    StateVectorPrimitive          -- abstract base class
    LearnedButterflyPermutation   -- learned Givens-rotation permutation
    UnifiedStateVectorEncoder     -- image -> state vector encoder
    MultiTaskBackbone             -- shared primitive-stack backbone
    MultiTaskModel                -- backbone + per-task heads
    StateVectorRMSNorm            -- per-state-vector normalization
    SparseDensePreconditioner     -- sparse+dense input preconditioner
    RotationalProjectionMixer     -- primary pairwise mixer (H_rotational)
    GramRoutedPermutation         -- Gram-based permutation (ablation)
    GramBlockDiagonalMixer        -- block-diagonal Gram mixer (ablation)
    SymmetricExpMixer             -- full exp(A) baseline mixer
"""

from models.primitives import StateVectorPrimitive, LearnedButterflyPermutation
from models.encoder import UnifiedStateVectorEncoder
from models.backbone import (
    MultiTaskBackbone,
    MultiTaskModel,
    StateVectorRMSNorm,
    SparseDensePreconditioner,
    RotationalProjectionMixer,
    GramRoutedPermutation,
    GramBlockDiagonalMixer,
)
from models.mixer import SymmetricExpMixer

__all__ = [
    "StateVectorPrimitive",
    "LearnedButterflyPermutation",
    "UnifiedStateVectorEncoder",
    "MultiTaskBackbone",
    "MultiTaskModel",
    "StateVectorRMSNorm",
    "SparseDensePreconditioner",
    "RotationalProjectionMixer",
    "GramRoutedPermutation",
    "GramBlockDiagonalMixer",
    "SymmetricExpMixer",
]
