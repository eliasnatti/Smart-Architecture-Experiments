"""
Backbone architecture components for multi-task geodesic experiment.

Contains:
  SparseDensePreconditioner    -- per-value blend of sparse and dense input forms
  RotationalProjectionMixer    -- pairwise mixer with A = theta*R_base + eps*A_perturb
  GramRoutedPermutation        -- state-vector permutation set from Gram matrix
  GramBlockDiagonalMixer       -- block-diagonal exp(A) with Gram-permuted features
  StateVectorRMSNorm           -- per-state-vector L2 normalization (post-block stability)
  MultiTaskBackbone            -- shared primitive-stack backbone over a 20x40 canvas
  TaskHead                     -- mean-pool + Linear readout per task
  MultiTaskModel               -- backbone + one TaskHead per task name
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.primitives import (
    StateVectorPrimitive,
    StateVectorPointwiseTransform,
    StateVectorFactorization,
    LearnedButterflyPermutation,
)
from models.encoder import UnifiedStateVectorEncoder
from models.mixer import SymmetricExpMixer

# ============================================================
# Canvas geometry constants (must match data.py)
# ============================================================

_MNIST_CROP_SIZE = 20
_CANVAS_H = _MNIST_CROP_SIZE          # 20
_CANVAS_W = _MNIST_CROP_SIZE * 2      # 40
_NUM_STATE_VECTORS = _CANVAS_H * _CANVAS_W  # 800
_BASE_DIM = 3                          # encoder outputs [intensity, x, y]
_DEFAULT_STATE_DIM = 12
_DEFAULT_DEPTH = 5

# ============================================================
# SparseDensePreconditioner
# ============================================================

class SparseDensePreconditioner(nn.Module):
    """Per-value blend of sparse and globally-mixed input forms.

    Operates on the (B, N, 3) output of UnifiedStateVectorEncoder.encode_mnist,
    BEFORE the lift to state_dim. The dense form is a per-batch 3x3 feature-
    correlation matrix applied per-pixel:

        G = (x_sparse^T @ x_sparse) / N        -- (B, 3, 3)
        x_dense = x_sparse @ G                 -- (B, N, 3)

    A1 and A2 are 3x3 block-diagonal: a 1x1 intensity scalar and a 2x2
    position block. Intensity and position get independent preconditioning
    coefficients.

    Total learnable params: 2 * (1 + 4) = 10.
    """

    def __init__(self):
        super().__init__()
        self.A1_intensity = nn.Parameter(torch.tensor(1.0))
        self.A1_position = nn.Parameter(torch.eye(2))
        self.A2_intensity = nn.Parameter(torch.tensor(0.0))
        self.A2_position = nn.Parameter(torch.zeros(2, 2))

    def _assemble(self, s_int, s_pos):
        """Block-diagonal 3x3: [[s_int, 0], [0, s_pos]]."""
        M = torch.zeros(3, 3, device=s_int.device, dtype=s_int.dtype)
        M = M.clone()
        M[0, 0] = s_int
        M[1:3, 1:3] = s_pos
        return M

    def forward(self, x_sparse):
        B, N, D = x_sparse.shape
        assert D == 3, f"SparseDensePreconditioner expects input dim 3, got {D}"

        A1 = self._assemble(self.A1_intensity, self.A1_position)
        A2 = self._assemble(self.A2_intensity, self.A2_position)

        G = torch.einsum('bnd,bne->bde', x_sparse, x_sparse) / float(N)   # (B,3,3)
        x_dense = torch.einsum('bnd,bde->bne', x_sparse, G)               # (B,N,3)

        out = x_sparse @ A1.t() + x_dense @ A2.t()
        return out

    @torch.no_grad()
    def density_signature(self):
        """Return (A1+A2) as a 3x3 tensor on CPU."""
        A1 = self._assemble(self.A1_intensity.detach(), self.A1_position.detach())
        A2 = self._assemble(self.A2_intensity.detach(), self.A2_position.detach())
        return (A1 + A2).cpu()


# ============================================================
# RotationalProjectionMixer
# ============================================================

class RotationalProjectionMixer(StateVectorPrimitive, nn.Module):
    """Pairwise mixer with A = theta * R_base + eps * A_perturb.

    Fixed canonical rotation generator R_base (skew-symmetric, 0 params)
    plus a low-rank learned perturbation. theta is predicted per (batch,
    pair) from the pair vector; eps is a global scalar.

    Two modes controlled by skew_only:
      True  (default): A_perturb = UV^T - VU^T (skew by construction).
            Full A is skew-symmetric => exp(A) is pure rotation.
      False : A_perturb = UV^T (general low-rank). Has both symmetric
            (magnitude scaling) and skew (rotation) components. Uses
            Pade [1,1] (Cayley transform) for speed.

    Exposes _last_A_full / _last_A_sym / _last_A_skew / _last_theta /
    _last_input / _last_output for analysis compatibility.
    """

    def __init__(self, num_state_vectors, state_dim, rank=2,
                 theta_head_scale=0.01, perturb_init_scale=0.01,
                 skew_only=True, use_matrix_exp=False):
        nn.Module.__init__(self)
        assert num_state_vectors % 2 == 0, \
            "num_state_vectors must be even (paired mixer)"
        self.num_state_vectors = num_state_vectors
        self.num_pairs = num_state_vectors // 2
        self.state_dim = state_dim
        self.pair_dim = 2 * state_dim
        self.rank = rank
        self.skew_only = skew_only
        self.use_matrix_exp = use_matrix_exp

        d = state_dim
        R_base = torch.zeros(self.pair_dim, self.pair_dim)
        R_base[:d, d:] = -torch.eye(d)
        R_base[d:, :d] = torch.eye(d)
        self.register_buffer('R_base', R_base)
        self.register_buffer('_I', torch.eye(self.pair_dim))

        self.theta_head = nn.Linear(self.pair_dim, 1)
        nn.init.normal_(self.theta_head.weight, std=theta_head_scale)
        nn.init.zeros_(self.theta_head.bias)

        self.U = nn.Parameter(torch.randn(self.pair_dim, rank) * perturb_init_scale)
        self.V = nn.Parameter(torch.randn(self.pair_dim, rank) * perturb_init_scale)
        self.eps = nn.Parameter(torch.tensor(0.0))

        self._last_A_full = None
        self._last_A_sym = None
        self._last_A_skew = None
        self._last_theta = None
        self._last_input = None
        self._last_output = None

    def _build_A_perturb(self):
        """Build perturbation matrix. (2d, 2d), shared across pairs."""
        if self.skew_only:
            return self.U @ self.V.t() - self.V @ self.U.t()
        else:
            return self.U @ self.V.t()

    def _pade11(self, A, pair_flat):
        """Pade [1,1] (Cayley) approximation: exp(A) ~= (I+A/2)(I-A/2)^-1."""
        half_A = A * 0.5
        I = self._I[None]                                # (1, 2d, 2d)
        lhs = I - half_A                                 # (B*P, 2d, 2d)
        rhs = torch.einsum('bij,bj->bi', I + half_A, pair_flat)  # (B*P, 2d)
        return torch.linalg.solve(lhs, rhs)              # (B*P, 2d)

    def forward_transform(self, state_vectors):
        """Apply exp(theta * R_base + eps * A_perturb) per pair."""
        B, N, d = state_vectors.shape
        assert d == self.state_dim, \
            f"state_dim mismatch: expected {self.state_dim}, got {d}"
        P = N // 2
        num_paired = P * 2

        s_i = state_vectors[:, 0:num_paired:2, :]        # (B, P, d)
        s_j = state_vectors[:, 1:num_paired:2, :]        # (B, P, d)
        pair_vectors = torch.cat([s_i, s_j], dim=2)      # (B, P, 2d)
        pair_flat = pair_vectors.reshape(B * P, self.pair_dim)

        theta = self.theta_head(pair_flat).squeeze(-1)   # (B*P,)

        A_perturb = self._build_A_perturb()              # (2d, 2d), shared
        A = (theta[:, None, None] * self.R_base[None]
             + self.eps * A_perturb[None])               # (B*P, 2d, 2d)

        self._last_A_full = A
        self._last_A_sym = (A + A.transpose(-2, -1)) / 2
        self._last_A_skew = (A - A.transpose(-2, -1)) / 2
        self._last_theta = theta
        self._last_input = pair_vectors

        if self.skew_only:
            with torch.no_grad():
                sym = self._last_A_sym
                sym_norm = sym.flatten(-2).norm(dim=-1)    # (B*P,)
                scale = torch.clamp(sym_norm / 3.0, min=1.0)
            A_safe = A / scale[:, None, None]
            M = torch.linalg.matrix_exp(A_safe)           # (B*P, 2d, 2d)
            mixed_flat = torch.einsum('bij,bj->bi', M, pair_flat)
        else:
            if self.use_matrix_exp:
                M = torch.linalg.matrix_exp(A)
                mixed_flat = torch.einsum('bij,bj->bi', M, pair_flat)
            else:
                mixed_flat = self._pade11(A, pair_flat)

        mixed = mixed_flat.view(B, P, self.pair_dim)
        self._last_output = mixed

        result = state_vectors.clone()
        result[:, 0:num_paired:2, :] = mixed[:, :, :d]
        result[:, 1:num_paired:2, :] = mixed[:, :, d:]
        return result

    def inverse_transform(self, state_vectors):
        raise NotImplementedError(
            "RotationalProjectionMixer.inverse_transform not implemented"
        )

    def is_invertible(self):
        return True

    def parameter_constraints(self):
        return (
            f"pairwise rotational mixer: A = theta * R_base + eps * A_perturb; "
            f"R_base fixed skew (0 params), A_perturb low-rank skew (rank={self.rank})"
        )

    def regularization_terms(self):
        """Penalty terms for the training loop."""
        device = self.eps.device
        if self._last_theta is None:
            theta_pull = torch.zeros((), device=device)
        else:
            theta_pull = torch.sin(2.0 * self._last_theta).pow(2).mean()
        eps_l2 = self.eps.pow(2)
        a_perturb_l1 = self._build_A_perturb().abs().mean()
        if self._last_A_sym is not None:
            a_sym_frob = self._last_A_sym.pow(2).sum(dim=(-2, -1)).mean()
        else:
            a_sym_frob = torch.zeros((), device=device)
        return {
            'theta_integer_pull': theta_pull,
            'eps_l2': eps_l2,
            'a_perturb_l1': a_perturb_l1,
            'a_sym_frob': a_sym_frob,
        }


# ============================================================
# GramRoutedPermutation
# ============================================================

class GramRoutedPermutation(StateVectorPrimitive, nn.Module):
    """State-vector permutation set from Gram matrix analysis.

    Starts as identity; call set_permutation() after warmup to install
    the Gram-based routing.
    """

    def __init__(self, num_state_vectors):
        nn.Module.__init__(self)
        self.num_state_vectors = num_state_vectors
        self.register_buffer('permutation',
                             torch.arange(num_state_vectors, dtype=torch.long))
        self.register_buffer('inverse_permutation',
                             torch.arange(num_state_vectors, dtype=torch.long))

    def set_permutation(self, perm):
        self.permutation.copy_(perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm), device=perm.device)
        self.inverse_permutation.copy_(inv)

    def forward_transform(self, state_vectors):
        return state_vectors[:, self.permutation, :]

    def inverse_transform(self, state_vectors):
        return state_vectors[:, self.inverse_permutation, :]

    def is_invertible(self):
        return True

    def parameter_constraints(self):
        return "Gram-routed permutation (fixed after warmup)"


# ============================================================
# GramBlockDiagonalMixer helpers
# ============================================================

def _equal_block_sizes(pair_dim, num_blocks):
    """Equal blocks; pair_dim distributed as evenly as possible."""
    base = pair_dim // num_blocks
    remainder = pair_dim % num_blocks
    return [base + (1 if i < remainder else 0) for i in range(num_blocks)]


class _SmallBlockMixer(nn.Module):
    """Rotational exp(A) mixer for one block within the block-diagonal."""

    def __init__(self, block_size, rank=2,
                 theta_head_scale=0.01, perturb_init_scale=0.01):
        super().__init__()
        self.block_size = block_size

        if block_size % 2 == 0:
            half = block_size // 2
            R = torch.zeros(block_size, block_size)
            R[:half, half:] = -torch.eye(half)
            R[half:, :half] = torch.eye(half)
        else:
            gen = torch.Generator().manual_seed(block_size)
            R = torch.randn(block_size, block_size, generator=gen)
            R = (R - R.t()) / 2
            R = R / R.norm() * math.sqrt(block_size)
        self.register_buffer('R_base', R)

        self.theta_head = nn.Linear(block_size, 1)
        nn.init.normal_(self.theta_head.weight, std=theta_head_scale)
        nn.init.zeros_(self.theta_head.bias)

        self.U = nn.Parameter(torch.randn(block_size, rank) * perturb_init_scale)
        self.V = nn.Parameter(torch.randn(block_size, rank) * perturb_init_scale)
        self.eps = nn.Parameter(torch.tensor(0.0))

        self._last_A = None
        self._last_theta = None

    def forward(self, x):
        """x: (BP, block_size) -> (BP, block_size)"""
        theta = self.theta_head(x).squeeze(-1)
        A_perturb = self.U @ self.V.t() - self.V @ self.U.t()
        A = (theta[:, None, None] * self.R_base[None]
             + self.eps * A_perturb[None])

        with torch.no_grad():
            sym = (A + A.transpose(-2, -1)) / 2
            sym_norm = sym.flatten(-2).norm(dim=-1)
            scale = torch.clamp(sym_norm / 3.0, min=1.0)
        A_safe = A / scale[:, None, None]

        M = torch.linalg.matrix_exp(A_safe)
        self._last_A = A
        self._last_theta = theta
        return torch.einsum('bij,bj->bi', M, x)


# ============================================================
# GramBlockDiagonalMixer
# ============================================================

class GramBlockDiagonalMixer(StateVectorPrimitive, nn.Module):
    """Pairwise mixer with Gram-permuted block-diagonal exp(A).

    Feature permutation (set by gram_warmup) reorders the pair_dim so
    correlated features are contiguous, then independent _SmallBlockMixers
    operate on variable-sized slices determined by the Gram eigenspectrum.

    Exposes _last_A_full / _last_A_sym / _last_A_skew / _last_input /
    _last_output for analysis-hook compatibility with compute_grassmannian_suite.
    """

    def __init__(self, num_state_vectors, state_dim, num_blocks=3, rank=2):
        nn.Module.__init__(self)
        assert num_state_vectors % 2 == 0
        self.num_state_vectors = num_state_vectors
        self.num_pairs = num_state_vectors // 2
        self.state_dim = state_dim
        self.pair_dim = 2 * state_dim
        self.rank = rank

        self.register_buffer('feature_perm',
                             torch.arange(self.pair_dim, dtype=torch.long))
        self.register_buffer('feature_perm_inv',
                             torch.arange(self.pair_dim, dtype=torch.long))

        block_sizes = _equal_block_sizes(self.pair_dim, num_blocks)
        self.block_sizes = block_sizes
        self.block_mixers = nn.ModuleList([
            _SmallBlockMixer(bs, rank=rank) for bs in block_sizes
        ])

        self._last_A_full = None
        self._last_A_sym = None
        self._last_A_skew = None
        self._last_theta = None
        self._last_input = None
        self._last_output = None

    @property
    def num_blocks(self):
        return len(self.block_sizes)

    def rebuild_blocks(self, block_sizes):
        """Replace block_mixers with new variable sizes."""
        assert sum(block_sizes) == self.pair_dim, \
            f"block_sizes sum {sum(block_sizes)} != pair_dim {self.pair_dim}"
        self.block_sizes = list(block_sizes)
        device = self.feature_perm.device
        self.block_mixers = nn.ModuleList([
            _SmallBlockMixer(bs, rank=self.rank) for bs in block_sizes
        ]).to(device)

    def set_feature_permutation(self, perm):
        """Install feature permutation from gram_warmup."""
        self.feature_perm.copy_(perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm), device=perm.device)
        self.feature_perm_inv.copy_(inv)

    def forward_transform(self, state_vectors):
        B, N, d = state_vectors.shape
        assert d == self.state_dim, \
            f"state_dim mismatch: expected {self.state_dim}, got {d}"
        P = N // 2
        num_paired = P * 2

        s_i = state_vectors[:, 0:num_paired:2, :]
        s_j = state_vectors[:, 1:num_paired:2, :]
        pair_vectors = torch.cat([s_i, s_j], dim=2)
        self._last_input = pair_vectors

        pair_flat = pair_vectors.reshape(B * P, self.pair_dim)

        pair_perm = pair_flat[:, self.feature_perm]

        blocks = pair_perm.split(self.block_sizes, dim=1)
        mixed_blocks = [bm(blk) for bm, blk in zip(self.block_mixers, blocks)]
        mixed_perm = torch.cat(mixed_blocks, dim=1)

        mixed_flat = mixed_perm[:, self.feature_perm_inv]
        mixed = mixed_flat.view(B, P, self.pair_dim)
        self._last_output = mixed

        BP = B * P
        A_block_diag = torch.zeros(BP, self.pair_dim, self.pair_dim,
                                   device=state_vectors.device)
        thetas = []
        offset = 0
        for bm in self.block_mixers:
            bs = bm.block_size
            if bm._last_A is not None:
                A_block_diag[:, offset:offset+bs, offset:offset+bs] = bm._last_A
            if bm._last_theta is not None:
                thetas.append(bm._last_theta)
            offset += bs

        fp = self.feature_perm
        A_orig = A_block_diag[:, fp][:, :, fp]

        self._last_A_full = A_orig
        self._last_A_sym = (A_orig + A_orig.transpose(-2, -1)) / 2
        self._last_A_skew = (A_orig - A_orig.transpose(-2, -1)) / 2
        self._last_theta = torch.cat(thetas) if thetas else None

        result = state_vectors.clone()
        result[:, 0:num_paired:2, :] = mixed[:, :, :d]
        result[:, 1:num_paired:2, :] = mixed[:, :, d:]
        return result

    def inverse_transform(self, state_vectors):
        raise NotImplementedError(
            "GramBlockDiagonalMixer.inverse_transform not implemented"
        )

    def is_invertible(self):
        return True

    def parameter_constraints(self):
        sizes_str = 'x'.join(str(s) for s in self.block_sizes)
        return (
            f"block-diagonal rotational mixer: {self.num_blocks} blocks "
            f"[{sizes_str}] within pair_dim={self.pair_dim}"
        )

    def regularization_terms(self):
        """Aggregate regularization across blocks."""
        device = self.block_mixers[0].eps.device
        theta_pull = torch.zeros((), device=device)
        eps_l2 = torch.zeros((), device=device)
        a_perturb_l1 = torch.zeros((), device=device)

        for bm in self.block_mixers:
            if bm._last_theta is not None:
                theta_pull = theta_pull + torch.sin(2.0 * bm._last_theta).pow(2).mean()
            eps_l2 = eps_l2 + bm.eps.pow(2)
            A_p = bm.U @ bm.V.t() - bm.V @ bm.U.t()
            a_perturb_l1 = a_perturb_l1 + A_p.abs().mean()

        n = self.num_blocks
        return {
            'theta_integer_pull': theta_pull / n,
            'eps_l2': eps_l2 / n,
            'a_perturb_l1': a_perturb_l1 / n,
        }


# ============================================================
# StateVectorRMSNorm
# ============================================================

class StateVectorRMSNorm(StateVectorPrimitive, nn.Module):
    """Per-state-vector L2 normalization to unit RMS. Zero learnable params.

    Stability adapter applied at the end of each primitive block
    (permute -> pointwise -> mixer -> rmsnorm). Guarantees the pair vector
    entering the next mixer has bounded norm, so exp(A) gains cannot
    compound across depth.

    Note: this is NOT an invertible primitive in the Paper 1 sense -- it's a
    strict projection onto the sphere of constant RMS. Kept in the primitive
    factorization list for plumbing simplicity but flagged as non-invertible.
    """

    def __init__(self, state_dim, eps=1e-6):
        nn.Module.__init__(self)
        self.state_dim = state_dim
        self.eps = eps
        self._target = math.sqrt(state_dim)

    def forward_transform(self, state_vectors):
        norm = state_vectors.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return state_vectors * (self._target / norm)

    def inverse_transform(self, state_vectors):
        return state_vectors

    def is_invertible(self):
        return False

    def parameter_constraints(self):
        return "non-invertible: projects state vectors onto sphere of fixed RMS"


# ============================================================
# MultiTaskBackbone
# ============================================================

class MultiTaskBackbone(nn.Module):
    """Shared primitive-stack backbone over a 20x40 canvas.

    encoder:  20x40 canvas -> (B, 800, 3)  [intensity, norm_x, norm_y]
    embed:    (B, 800, 3)  -> (B, 800, state_dim)  zero-pad into Gr(3, state_dim)
    stack:    depth x (permute -> pointwise -> mixer -> RMSNorm)
    output:   (B, 800, state_dim) -- NOT pooled; task heads handle pooling

    backbone_kind options:
      'baseline'   -- SymmetricExpMixer (full matrix exp, no preconditioner)
      'rotational' -- RotationalProjectionMixer + SparseDensePreconditioner
      'gram_block' -- GramBlockDiagonalMixer + GramRoutedPermutation + preconditioner
    """

    def __init__(self, depth=_DEFAULT_DEPTH, state_dim=_DEFAULT_STATE_DIM,
                 backbone_kind='rotational', num_blocks=1,
                 mixer_skew_only=True, mixer_use_matrix_exp=False):
        super().__init__()
        assert backbone_kind in ('baseline', 'rotational', 'gram_block'), \
            f"unknown backbone_kind: {backbone_kind}"
        self.state_dim = state_dim
        self.depth = depth
        self.backbone_kind = backbone_kind
        self.num_blocks = num_blocks

        self.encoder = UnifiedStateVectorEncoder(
            image_height=_CANVAS_H, image_width=_CANVAS_W, state_dim=_BASE_DIM
        )

        if backbone_kind in ('rotational', 'gram_block'):
            self.preconditioner = SparseDensePreconditioner()
        else:
            self.preconditioner = None

        primitives = []
        self.mixers = []
        self.butterflies = []
        for d_idx in range(depth):
            if backbone_kind == 'gram_block':
                primitives.append(GramRoutedPermutation(_NUM_STATE_VECTORS))
            else:
                bf = LearnedButterflyPermutation(
                    _NUM_STATE_VECTORS, stage=d_idx,
                )
                primitives.append(bf)
                self.butterflies.append(bf)

            primitives.append(StateVectorPointwiseTransform(
                num_state_vectors=_NUM_STATE_VECTORS, state_dim=state_dim,
                transform_type='shared', initialization='near_identity'
            ))

            if backbone_kind == 'gram_block':
                mixer = GramBlockDiagonalMixer(
                    num_state_vectors=_NUM_STATE_VECTORS, state_dim=state_dim,
                    num_blocks=num_blocks
                )
            elif backbone_kind == 'rotational':
                mixer = RotationalProjectionMixer(
                    num_state_vectors=_NUM_STATE_VECTORS, state_dim=state_dim,
                    skew_only=mixer_skew_only,
                    use_matrix_exp=mixer_use_matrix_exp,
                )
            else:
                mixer = SymmetricExpMixer(
                    num_state_vectors=_NUM_STATE_VECTORS, state_dim=state_dim
                )
            primitives.append(mixer)
            self.mixers.append(mixer)

            primitives.append(StateVectorRMSNorm(state_dim=state_dim))

        self.primitive_factorization = StateVectorFactorization(primitives)

    def forward(self, canvas, capture_intermediates=False):
        """canvas: (B, 1, 20, 40) -> (B, 800, state_dim)

        If capture_intermediates=True, also returns a list of per-layer
        post-mixer activations (before RMSNorm) as (B, N, state_dim)
        grad-connected tensors.
        """
        state_vectors = self.encoder.encode_mnist(canvas)   # (B, 800, 3)
        if self.preconditioner is not None:
            state_vectors = self.preconditioner(state_vectors)
        B, N, d = state_vectors.shape
        padded = state_vectors.new_zeros(B, N, self.state_dim)
        padded[:, :, :d] = state_vectors
        sv = padded

        if not capture_intermediates:
            return self.primitive_factorization.forward_transform(sv)

        prims = self.primitive_factorization.primitives
        prims_per_block = 4  # perm, pointwise, mixer, rmsnorm
        layer_acts = []
        for l in range(self.depth):
            base = prims_per_block * l
            sv = prims[base].forward_transform(sv)       # permute
            sv = prims[base + 1].forward_transform(sv)   # pointwise
            sv = prims[base + 2].forward_transform(sv)   # mixer
            layer_acts.append(sv)                         # post-mix, pre-norm
            sv = prims[base + 3].forward_transform(sv)   # rmsnorm
        return sv, layer_acts


# ============================================================
# TaskHead
# ============================================================

class TaskHead(nn.Module):
    """Mean-pool over state vectors then Linear to task output dim."""

    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(state_dim, output_dim)

    def forward(self, features):
        pooled = features.mean(dim=1)
        return self.linear(pooled)


# ============================================================
# MultiTaskModel
# ============================================================

class MultiTaskModel(nn.Module):
    """Backbone + one TaskHead per task name."""

    # Import TASK_SPECS lazily to avoid circular import at module load time.
    # The actual dict is accessed from data.TASK_SPECS inside __init__.

    def __init__(self, task_names, depth=_DEFAULT_DEPTH, state_dim=_DEFAULT_STATE_DIM,
                 backbone_kind='rotational', num_blocks=1,
                 mixer_skew_only=True, mixer_use_matrix_exp=False):
        super().__init__()
        from data import TASK_SPECS  # imported here to avoid circular import
        self.backbone = MultiTaskBackbone(
            depth=depth, state_dim=state_dim, backbone_kind=backbone_kind,
            num_blocks=num_blocks,
            mixer_skew_only=mixer_skew_only,
            mixer_use_matrix_exp=mixer_use_matrix_exp,
        )
        self.heads = nn.ModuleDict({
            name: TaskHead(state_dim, TASK_SPECS[name]['output_dim'])
            for name in task_names
        })

    def forward_task(self, canvas, task_name, capture_intermediates=False):
        if capture_intermediates:
            features, layer_acts = self.backbone(canvas, capture_intermediates=True)
            return self.heads[task_name](features), layer_acts
        features = self.backbone(canvas)
        return self.heads[task_name](features)
