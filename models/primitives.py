"""
State vector primitive ABC and learned butterfly permutation.

StateVectorPrimitive: abstract base class for all primitives.
StateVectorPointwiseTransform: G_i in GL(d) per state vector.
StateVectorFactorization: sequential composition of primitives.
LearnedButterflyPermutation: soft Givens-rotation butterfly stage.
"""

import math
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


# ============================================================
# Abstract base class
# ============================================================

class StateVectorPrimitive(ABC):
    """Abstract base class for state vector primitives.

    Primitives operate on (batch_size, N, d) tensors where each
    row s_i in R^d is a unified position-feature state vector.
    """

    @abstractmethod
    def forward_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        """Apply transformation to state vectors.

        Args:
            state_vectors: (batch_size, N, d)
        Returns:
            transformed: (batch_size, N, d)
        """

    @abstractmethod
    def inverse_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformation to state vectors."""

    @abstractmethod
    def is_invertible(self) -> bool:
        """Return True if the current parameters make this primitive invertible."""

    @abstractmethod
    def parameter_constraints(self) -> str:
        """Return a string describing the invertibility constraints."""


# ============================================================
# Pointwise transform
# ============================================================

class StateVectorPointwiseTransform(StateVectorPrimitive, nn.Module):
    """Pointwise transform: G_i in GL(d) applied to each state vector.

    transform_type='shared': all state vectors share one d x d matrix.
    transform_type='individual': each state vector has its own matrix.
    """

    def __init__(self, num_state_vectors: int, state_dim: int,
                 transform_type: str = "shared",
                 initialization: str = "near_identity"):
        nn.Module.__init__(self)
        self.num_state_vectors = num_state_vectors
        self.state_dim = state_dim
        self.transform_type = transform_type
        self.initialization = initialization

        if transform_type == "shared":
            self.transforms = nn.Parameter(self._init_matrix(state_dim, state_dim))
            self.num_transforms = 1
        elif transform_type == "individual":
            self.transforms = nn.Parameter(
                self._init_matrix(num_state_vectors, state_dim, state_dim)
            )
            self.num_transforms = num_state_vectors
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

    def _init_matrix(self, *shape) -> torch.Tensor:
        matrix = torch.zeros(*shape)
        if self.initialization == "identity":
            if len(shape) == 2:
                matrix = torch.eye(shape[0])
            else:
                for i in range(shape[0]):
                    matrix[i] = torch.eye(shape[1])
        elif self.initialization == "near_identity":
            if len(shape) == 2:
                matrix = torch.eye(shape[0]) + 0.01 * torch.randn(*shape)
            else:
                for i in range(shape[0]):
                    matrix[i] = torch.eye(shape[1]) + 0.01 * torch.randn(shape[1], shape[2])
        elif self.initialization == "orthogonal":
            if len(shape) == 2:
                matrix = torch.nn.init.orthogonal_(torch.empty(*shape))
            else:
                for i in range(shape[0]):
                    matrix[i] = torch.nn.init.orthogonal_(torch.empty(shape[1], shape[2]))
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")
        return matrix

    def forward_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "shared":
            return torch.einsum('bnd,de->bne', state_vectors, self.transforms)
        elif self.transform_type == "individual":
            return torch.einsum('bnd,nde->bne', state_vectors, self.transforms)

    def inverse_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "shared":
            G_inv = torch.inverse(self.transforms)
            return torch.einsum('bnd,de->bne', state_vectors, G_inv)
        elif self.transform_type == "individual":
            G_inv = torch.inverse(self.transforms)
            return torch.einsum('bnd,nde->bne', state_vectors, G_inv)

    def is_invertible(self) -> bool:
        try:
            if self.transform_type == "shared":
                return torch.abs(torch.det(self.transforms)).item() > 1e-6
            else:
                dets = torch.det(self.transforms)
                return bool(torch.all(torch.abs(dets) > 1e-6).item())
        except Exception:
            return False

    def parameter_constraints(self) -> str:
        return f"All {self.state_dim}x{self.state_dim} matrices must have det != 0"


# ============================================================
# Factorization
# ============================================================

class StateVectorFactorization(nn.Module):
    """Sequential composition of state vector primitives."""

    def __init__(self, primitives: List[StateVectorPrimitive]):
        super().__init__()
        self.primitives = primitives
        self._module_primitives = nn.ModuleList([
            p for p in primitives if isinstance(p, nn.Module)
        ])

    def forward_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        result = state_vectors
        for primitive in self.primitives:
            result = primitive.forward_transform(result)
        return result

    def inverse_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        result = state_vectors
        for primitive in reversed(self.primitives):
            result = primitive.inverse_transform(result)
        return result

    def is_invertible(self) -> bool:
        return all(p.is_invertible() for p in self.primitives)

    def __len__(self):
        return len(self.primitives)


# ============================================================
# Learned butterfly permutation
# ============================================================

class LearnedButterflyPermutation(StateVectorPrimitive, nn.Module):
    """Learned butterfly permutation: soft Givens rotation per swap pair.

    Replaces the fixed hard-swap butterfly stage with a learnable rotation
    angle phi per pair. phi=0: identity (no exchange), phi=pi/2: full swap.

    Forward per pair (i, j):
        s_i_out =  cos(phi) * s_i + sin(phi) * s_j
        s_j_out = -sin(phi) * s_i + cos(phi) * s_j

    This is a Givens rotation in the (i, j) plane -- orthogonal by
    construction, invertible (inverse uses -phi).

    Params per layer: num_pairs (~N/2).
    """

    def __init__(self, num_state_vectors: int, stage: int = 0,
                 init_phi: float = math.pi / 2):
        nn.Module.__init__(self)
        self.num_state_vectors = num_state_vectors
        self.stage = stage

        log_n = max(1, int(math.log2(num_state_vectors) + 1))
        distance = 2 ** (stage % log_n)
        pairs_i = []
        pairs_j = []
        for i in range(num_state_vectors):
            if (i // distance) % 2 == 0:
                j = i + distance
                if j < num_state_vectors:
                    pairs_i.append(i)
                    pairs_j.append(j)

        self.register_buffer('pairs_i', torch.tensor(pairs_i, dtype=torch.long))
        self.register_buffer('pairs_j', torch.tensor(pairs_j, dtype=torch.long))
        self.num_pairs = len(pairs_i)
        self.distance = distance

        self.phi = nn.Parameter(torch.full((self.num_pairs,), init_phi))
        self._last_phi = None

    def forward_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)
        self._last_phi = self.phi

        s_i = state_vectors[:, self.pairs_i, :]
        s_j = state_vectors[:, self.pairs_j, :]

        c = cos_phi[None, :, None]
        s = sin_phi[None, :, None]

        new_i = c * s_i + s * s_j
        new_j = -s * s_i + c * s_j

        result = state_vectors.clone()
        result[:, self.pairs_i, :] = new_i
        result[:, self.pairs_j, :] = new_j
        return result

    def inverse_transform(self, state_vectors: torch.Tensor) -> torch.Tensor:
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)

        s_i = state_vectors[:, self.pairs_i, :]
        s_j = state_vectors[:, self.pairs_j, :]

        c = cos_phi[None, :, None]
        s = sin_phi[None, :, None]

        new_i = c * s_i - s * s_j
        new_j = s * s_i + c * s_j

        result = state_vectors.clone()
        result[:, self.pairs_i, :] = new_i
        result[:, self.pairs_j, :] = new_j
        return result

    def is_invertible(self) -> bool:
        return True

    def parameter_constraints(self) -> str:
        return (f"learned butterfly stage {self.stage}: "
                f"{self.num_pairs} Givens rotations, distance={self.distance}")

    def routing_stats(self) -> dict:
        """Summary statistics for analysis logging."""
        with torch.no_grad():
            phi_abs = self.phi.abs()
            return {
                'mean_phi': phi_abs.mean().item(),
                'active': (phi_abs > 0.1).float().mean().item(),
                'full_swap': ((phi_abs - math.pi / 2).abs() < 0.1).float().mean().item(),
            }
