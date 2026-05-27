"""
SymmetricExpMixer: data-dependent pairwise mixer with full exp(A).

Extracted from exp6_geometric_loss.py for use as the baseline architecture
in the multi-task geodesic experiment. The mixer computes A = mix_net(pair)
reshaped to (2d, 2d), then applies M = exp(A) to each pair.

SymmetricExpMixer is a backward-compatible alias for ExpMixer (the class was
renamed when A was extended to full A = S + K; the alias keeps downstream
imports working unchanged).

Geometric regularization helpers (compute_geometric_reg, compute_frobenius_reg)
are included for training loops that use them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.primitives import StateVectorPrimitive


# ============================================================
# Base pairwise mixer (minimal extraction from state_vector_primitives)
# ============================================================

class _StateVectorPairwiseMixerBase(StateVectorPrimitive, nn.Module):
    """Minimal base for data-dependent pairwise mixers.

    Provides the mix_net (Linear -> pair-shaped A matrix) and the num_pairs
    bookkeeping that ExpMixer's forward_transform relies on.
    """

    def __init__(self, num_state_vectors, state_dim,
                 weight_mode='data_dependent', weight_sharing='shared',
                 mixing_strength=0.1):
        nn.Module.__init__(self)
        assert num_state_vectors % 2 == 0, \
            "num_state_vectors must be even"
        self.num_state_vectors = num_state_vectors
        self.num_pairs = num_state_vectors // 2
        self.state_dim = state_dim
        self.matrix_dim = 2 * state_dim

        # Data-dependent mix_net: Linear(2d, (2d)^2)
        self.mix_net = nn.Linear(self.matrix_dim, self.matrix_dim * self.matrix_dim)
        nn.init.normal_(self.mix_net.weight, std=mixing_strength)
        nn.init.zeros_(self.mix_net.bias)

    # StateVectorPrimitive ABC stubs (concrete class overrides forward_transform)
    def forward_transform(self, state_vectors):
        raise NotImplementedError

    def inverse_transform(self, state_vectors):
        raise NotImplementedError("Pairwise exp(A) mixer inverse not implemented")

    def is_invertible(self):
        return True  # exp(A) is always invertible

    def parameter_constraints(self):
        return "pairwise exp(A) mixer: always invertible for finite A"

    def regularization_terms(self):
        """Compatibility stub -- returns zero terms.

        ExpMixer stores _last_A_sym after each forward pass; callers can call
        this method and receive zero tensors before the first forward.
        """
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        A = getattr(self, '_last_A_sym', None)
        if A is not None:
            a_sym_frob = A.pow(2).sum(dim=(-2, -1)).mean()
        else:
            a_sym_frob = zero
        A_full = getattr(self, '_last_A_full', None)
        if A_full is not None:
            a_perturb_l1 = A_full.abs().mean()
        else:
            a_perturb_l1 = zero
        return {
            'a_sym_frob': a_sym_frob,
            'a_perturb_l1': a_perturb_l1,
        }


# ============================================================
# ExpMixer (full A = S + K)
# ============================================================

class ExpMixer(_StateVectorPairwiseMixerBase):
    """Data-dependent pairwise mixer with full A = S + K.

    Architecture:
      A = mix_net(pair_vector) reshaped to (2d, 2d)     # full, arbitrary
      M = exp(A)

    A varies per (batch, pair) so the full A tensor is (B*P, 2d, 2d).
    Per-forward grad-connected storage:
      _last_A_full  : full A  (for general-purpose analysis)
      _last_A_sym   : (A + A^T) / 2 -- S-component (backward compat for eigvalsh loggers)
      _last_A_skew  : (A - A^T) / 2 -- K-component (rotation analysis)

    Activation stability is handled outside this mixer by a post-block
    normalization layer (StateVectorRMSNorm in backbone.py).
    """

    def __init__(self, num_state_vectors, state_dim=3, mixing_strength=0.1):
        super().__init__(
            num_state_vectors=num_state_vectors,
            state_dim=state_dim,
            weight_mode='data_dependent',
            weight_sharing='shared',
            mixing_strength=mixing_strength,
        )
        self._last_A_full = None
        self._last_A_sym = None
        self._last_A_skew = None
        self._last_input = None
        self._last_output = None

    def forward_transform(self, state_vectors):
        """Apply per-pair exp(A) to consecutive pairs. Full A = S + K."""
        batch_size, N, d = state_vectors.shape
        matrix_dim = 2 * d
        P = self.num_pairs
        num_paired = P * 2

        # Extract pairs
        paired_states = state_vectors[:, :num_paired, :]
        s_i = paired_states[:, 0::2, :]  # (B, P, d)
        s_j = paired_states[:, 1::2, :]  # (B, P, d)
        pair_vectors = torch.cat([s_i, s_j], dim=2)  # (B, P, 2d)
        pair_flat = pair_vectors.reshape(batch_size * P, matrix_dim)

        # Data-dependent full A = S + K (no symmetric projection)
        A_flat = self.mix_net(pair_flat)                   # (B*P, (2d)^2)
        A = A_flat.view(batch_size * P, matrix_dim, matrix_dim)

        # Store grad-connected for geo_reg backprop and per-component analysis.
        self._last_A_full = A
        self._last_A_sym = (A + A.transpose(-2, -1)) / 2   # S-component
        self._last_A_skew = (A - A.transpose(-2, -1)) / 2  # K-component
        self._last_input = pair_vectors

        # Matrix exponential per pair with spectral radius guard on real part.
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(A)                      # complex
            real_radius = eigvals.real.abs().max(dim=-1).values    # (B*P,)
            scale = torch.clamp(real_radius / 3.0, min=1.0)        # exp(3)~20
        A_safe = A / scale[:, None, None]
        M = torch.linalg.matrix_exp(A_safe)                # (B*P, 2d, 2d)

        # Apply and reshape
        mixed_pairs = torch.einsum('bij,bj->bi', M, pair_flat)
        mixed_pairs = mixed_pairs.view(batch_size, P, matrix_dim)

        self._last_output = mixed_pairs

        # Reconstruct
        result = state_vectors.clone()
        result[:, 0:num_paired:2, :] = mixed_pairs[:, :, :d]
        result[:, 1:num_paired:2, :] = mixed_pairs[:, :, d:]
        return result


# Backward-compat alias: existing imports use `SymmetricExpMixer`.
# The class is now full-A; the old name maps to the new one so nothing breaks.
SymmetricExpMixer = ExpMixer


# ============================================================
# Geometric regularization helpers
# ============================================================

def compute_geometric_reg(mixers, device='cpu'):
    """Compute per-dimension geometric loss across all mixer layers.

    For each mixer, compute principal angles between input and output
    top-k subspaces on Gr(k, 2d) with k = d (half the ambient dimension).
    Returns the max sin^2(theta) across all layers and dimensions.

    Args:
        mixers: list of ExpMixer/SymmetricExpMixer instances
        device: torch device string (for zero tensor fallback)
    Returns:
        Scalar tensor: max sin^2(principal angle) across all layers.
    """
    max_dim_loss = torch.zeros((), device=device)

    for mixer in mixers:
        x_in = getattr(mixer, '_last_input', None)
        x_out = getattr(mixer, '_last_output', None)
        if x_in is None or x_out is None:
            continue

        # x_in, x_out: (B, P, 2d) -- flatten to (B*P, 2d)
        B, P, D = x_in.shape
        flat_in = x_in.reshape(B * P, D).float()
        flat_out = x_out.reshape(B * P, D).float()

        # k = half ambient dimension -> nontrivial Grassmannian distance
        U_in, _, _ = torch.linalg.svd(flat_in.T, full_matrices=False)
        U_out, _, _ = torch.linalg.svd(flat_out.T, full_matrices=False)
        k = D // 2
        k = min(k, U_in.shape[1], U_out.shape[1])
        cos_angles = torch.linalg.svdvals(U_in[:, :k].T @ U_out[:, :k])
        cos_sq = cos_angles.clamp(-1.0, 1.0) ** 2
        dim_loss = (1.0 - cos_sq).clamp(min=0.0)

        layer_max = dim_loss.max()
        max_dim_loss = torch.max(max_dim_loss, layer_max)

    return max_dim_loss


def compute_frobenius_reg(mixers, device='cpu'):
    """Compute ||A||_F^2 penalty summed over all mixer layers.

    Unlike sin^2(theta) (blind to eigenvalue magnitude), the Frobenius
    norm squared equals sum_i |lambda_i|^2 for symmetric A. Penalizing
    it bounds the spectral radius and prevents exp(A) blow-up.

    Uses the grad-connected _last_A_sym tensor stored on each mixer, so
    gradients flow back to mix_net.weight.

    Args:
        mixers: list of ExpMixer/SymmetricExpMixer instances
        device: torch device string
    Returns:
        Scalar tensor: mean ||A_sym||_F^2 averaged over layers.
    """
    total = torch.zeros((), device=device)
    count = 0
    for mixer in mixers:
        A = getattr(mixer, '_last_A_sym', None)
        if A is None:
            continue
        # A: (B*P, 2d, 2d) -- ||A||_F^2 averaged over batch*pair axis
        total = total + (A ** 2).sum(dim=(-2, -1)).mean()
        count += 1
    if count == 0:
        return torch.zeros((), device=device)
    return total / count
