"""
Grassmannian analysis suite for the multi-task geodesic experiment.

Functions for capturing layer activations, computing subspace geometry, and
measuring the perpendicular reservoir rank.

All measurements are post-hoc (no gradient required) and operate on saved
model checkpoints.
"""

import torch
import numpy as np

from data import (
    TASK_SPECS, TRAINED_TASKS, HELD_OUT_TASK,
    prepare_task_batch, make_canvas_pair, crop_mnist, build_loaders,
)
from models.backbone import MultiTaskBackbone, MultiTaskModel, TaskHead

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFER_EPOCHS = 10
TRANSFER_LR = 1e-3


# ============================================================
# Activation capture
# ============================================================

def capture_layer_activations(model, loaders, num_samples=512):
    """Run the backbone on a fixed canvas batch and record per-primitive outputs.

    Returns dict keyed by layer index:
        acts[l]['post_permute']: (num_samples, num_sv, state_dim)
        acts[l]['post_mix']:     (num_samples, num_sv, state_dim)

    Uses the single_test loader with the single-image canvas layout.
    post_mix is captured pre-RMSNorm so the geometric measurement reflects
    the mixer's direct output.
    """
    model.eval()
    backbone = model.backbone
    prims = backbone.primitive_factorization.primitives
    depth = backbone.depth
    prims_per_block = 4  # perm, pointwise, mixer, rmsnorm

    layer_acts = {
        l: {'post_permute': [], 'post_mix': []}
        for l in range(depth)
    }

    def _probe_forward(canvas):
        sv = backbone.encoder.encode_mnist(canvas)
        if backbone.preconditioner is not None:
            sv = backbone.preconditioner(sv)
        B, N, d = sv.shape
        padded = sv.new_zeros(B, N, backbone.state_dim)
        padded[:, :, :d] = sv
        sv = padded
        for l in range(depth):
            base = prims_per_block * l
            sv = prims[base].forward_transform(sv)
            layer_acts[l]['post_permute'].append(sv.detach().cpu())
            sv = prims[base + 1].forward_transform(sv)
            sv = prims[base + 2].forward_transform(sv)
            layer_acts[l]['post_mix'].append(sv.detach().cpu())
            sv = prims[base + 3].forward_transform(sv)
        return sv

    gathered = 0
    with torch.no_grad():
        for raw in loaders['single_test']:
            canvas, _ = prepare_task_batch('classification', raw, device=DEVICE)
            _probe_forward(canvas)
            gathered += canvas.size(0)
            if gathered >= num_samples:
                break

    out = {}
    for l in range(depth):
        out[l] = {
            'post_permute': torch.cat(layer_acts[l]['post_permute'], dim=0)[:num_samples],
            'post_mix':     torch.cat(layer_acts[l]['post_mix'], dim=0)[:num_samples],
        }
    return out


# ============================================================
# Subspace geometry
# ============================================================

def orthonormal_subspace(activations, k):
    """Return a (state_dim, k) orthonormal basis for the top-k singular
    directions of the activations.

    activations: (N, num_sv, state_dim) -> flatten to (N*num_sv, state_dim)
    """
    N, M, d = activations.shape
    flat = activations.reshape(N * M, d).float()
    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    V = Vh[:k].T   # (d, k)
    return V


def head_subspace(head_linear, output_dim):
    """Orthonormal basis for the column space of the head weight.

    head_linear.weight: (output_dim, state_dim)
    Returns: (state_dim, min(state_dim, output_dim))
    """
    W = head_linear.weight.detach().cpu().float()   # (out, d)
    U, S, Vh = torch.linalg.svd(W.T, full_matrices=False)
    k = min(W.T.shape[0], W.T.shape[1])
    return U[:, :k]  # (d, k)


def principal_angles(V_a, V_b):
    """Return sorted cosines of principal angles between two subspaces.

    V_a: (d, k_a), V_b: (d, k_b). Both expected orthonormal.
    Returns: singular values of V_a^T V_b, clamped to [-1, 1].
    """
    M = V_a.T @ V_b
    s = torch.linalg.svdvals(M)
    s = s.clamp(-1.0, 1.0)
    return s


def latent_occupancy(layer_V, V_ref):
    """Grassmannian distance as mean sin^2(theta).

    A value of 0 means V_ell == V_ref; values approaching 1 mean V_ell is
    in the orthogonal complement of V_ref (fully latent / in transit).
    """
    cos = principal_angles(layer_V, V_ref)
    sin_sq = (1.0 - cos ** 2).clamp(min=0.0)
    return float(sin_sq.mean().item())


def meaning_fraction(layer_V, V_out):
    """Per-task meaning fraction mu_ell.

        mu_ell = mean(cos^2(theta_j))

    A layer whose subspace is already V_out has mu_ell = 1;
    an orthogonal layer has mu_ell = 0.
    """
    cos = principal_angles(layer_V, V_out)
    return float((cos ** 2).mean().item())


# ============================================================
# Full Grassmannian suite
# ============================================================

def compute_grassmannian_suite(model, heads_by_task, loaders, num_samples=512, k=None):
    """Full Grassmannian measurement suite for one trained model.

    Returns dict:
        'layer_occupancy':           list[float] len=depth
        'meaning_per_task':          {task: [mu_ell for l in depth]}
        'reservoir_effective_rank':  scalar
        'per_primitive_occupancy':   {'post_permute': [...], 'post_mix': [...]}
        'mixer_eigenvalue_spectrum': list[dict] per layer
    """
    if k is None:
        k = model.backbone.state_dim // 2

    acts = capture_layer_activations(model, loaders, num_samples=num_samples)

    V_in = orthonormal_subspace(acts[0]['post_permute'], k)

    V_out_per_task = {
        t: head_subspace(h.linear, TASK_SPECS[t]['output_dim'])
        for t, h in heads_by_task.items()
    }

    depth = model.backbone.depth

    layer_occupancy = []
    per_prim_occ = {'post_permute': [], 'post_mix': []}
    meaning_per_task = {t: [] for t in V_out_per_task}

    for l in range(depth):
        V_perm = orthonormal_subspace(acts[l]['post_permute'], k)
        V_mix = orthonormal_subspace(acts[l]['post_mix'], k)

        V_ell = V_mix
        layer_occupancy.append(latent_occupancy(V_ell, V_in))

        per_prim_occ['post_permute'].append(latent_occupancy(V_perm, V_in))
        per_prim_occ['post_mix'].append(latent_occupancy(V_mix, V_in))

        for t, V_out in V_out_per_task.items():
            meaning_per_task[t].append(meaning_fraction(V_ell, V_out))

    # Reservoir rank: effective rank of stacked perpendicular components
    probe_V = orthonormal_subspace(acts[depth - 1]['post_mix'], k)
    perps = []
    for t, V_out in V_out_per_task.items():
        proj = probe_V @ (probe_V.T @ V_out)
        perp = V_out - proj
        perps.append(perp)
    if perps:
        stacked = torch.cat(perps, dim=1)
        sv = torch.linalg.svdvals(stacked)
        s2 = sv ** 2
        s2_norm = s2 / (s2.sum() + 1e-12)
        entropy = -(s2_norm * torch.log(s2_norm + 1e-12)).sum()
        eff_rank = float(torch.exp(entropy).item())
    else:
        eff_rank = 0.0

    # Mixer eigenvalue spectrum
    mixer_eig_stats = []
    model.eval()
    with torch.no_grad():
        sample_raw = next(iter(loaders['single_test']))
        sample_canvas, _ = prepare_task_batch('classification', sample_raw,
                                              device=DEVICE)
        model.backbone(sample_canvas)  # populates _last_A_* on each mixer

        for l, mixer in enumerate(model.backbone.mixers):
            A_full = getattr(mixer, '_last_A_full', None)
            if A_full is None:
                mixer_eig_stats.append({
                    'sym_effective_rank': 0.0, 'sym_eigenvalues': [],
                    'skew_effective_rank': 0.0, 'skew_eigenvalues': [],
                    'effective_rank': 0.0, 'eigenvalues': [],
                })
                continue

            stats = {}
            for component, name in [('_last_A_sym', 'sym'), ('_last_A_skew', 'skew')]:
                A = getattr(mixer, component, None)
                if A is None:
                    stats[f'{name}_effective_rank'] = 0.0
                    stats[f'{name}_eigenvalues'] = []
                    continue
                eigvals = torch.linalg.eigvalsh(A)
                mean_abs = eigvals.abs().mean(dim=0)
                e2 = mean_abs ** 2
                e2_norm = e2 / (e2.sum() + 1e-12)
                ent = -(e2_norm * torch.log(e2_norm + 1e-12)).sum()
                stats[f'{name}_effective_rank'] = float(torch.exp(ent).item())
                stats[f'{name}_eigenvalues'] = mean_abs.tolist()

            eigvals_full = torch.linalg.eigvals(A_full)
            mean_abs_full = eigvals_full.abs().mean(dim=0)
            e2f = mean_abs_full ** 2
            e2f_norm = e2f / (e2f.sum() + 1e-12)
            ent_f = -(e2f_norm * torch.log(e2f_norm + 1e-12)).sum()
            stats['effective_rank'] = float(torch.exp(ent_f.real).item())
            stats['eigenvalues'] = mean_abs_full.tolist()

            mixer_eig_stats.append(stats)

    return {
        'layer_occupancy': layer_occupancy,
        'meaning_per_task': meaning_per_task,
        'reservoir_effective_rank': eff_rank,
        'per_primitive_occupancy': per_prim_occ,
        'mixer_eigenvalue_spectrum': mixer_eig_stats,
    }


# ============================================================
# Transfer evaluation
# ============================================================

def transfer_to_task6(backbone, loaders, seed,
                      epochs=TRANSFER_EPOCHS, verbose=True):
    """Freeze backbone, train a fresh head for the held-out magnitude bucket task."""
    import torch.optim as optim
    from data import HELD_OUT_TASK

    torch.manual_seed(seed + 1000)

    for p in backbone.parameters():
        p.requires_grad_(False)

    head = TaskHead(backbone.state_dim, TASK_SPECS[HELD_OUT_TASK]['output_dim']).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=TRANSFER_LR)

    # Wrap into a temporary module for eval_task compatibility
    wrapper = torch.nn.Module()
    wrapper.backbone = backbone
    wrapper.heads = torch.nn.ModuleDict({HELD_OUT_TASK: head})

    def _forward_task(canvas, task_name):
        assert task_name == HELD_OUT_TASK
        feats = backbone(canvas)
        return head(feats)
    wrapper.forward_task = _forward_task

    best_acc = 0.0
    import time as _time
    for epoch in range(epochs):
        ep_t0 = _time.time()
        head.train()
        losses = []
        for raw in loaders['single_train']:
            canvas, target = prepare_task_batch(HELD_OUT_TASK, raw, device=DEVICE)
            optimizer.zero_grad()
            logits = head(backbone(canvas))
            from training import task_loss as _task_loss
            loss = _task_loss(HELD_OUT_TASK, logits, target)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        m = _eval_task_wrapper(wrapper, HELD_OUT_TASK, loaders)
        if m['accuracy'] > best_acc:
            best_acc = m['accuracy']
        ep_sec = _time.time() - ep_t0
        train_loss = float(np.mean(losses)) if losses else float('nan')
        if verbose:
            print(f"    transfer {epoch+1}/{epochs}: "
                  f"loss={train_loss:.4f}, acc={m['accuracy']:.3f}, "
                  f"best={best_acc:.3f} [{ep_sec:.0f}s]")

    for p in backbone.parameters():
        p.requires_grad_(True)

    return {'best_accuracy': best_acc, 'final_accuracy': m['accuracy']}


def _eval_task_wrapper(wrapper, task_name, loaders):
    """eval_task for a plain wrapper module (not MultiTaskModel)."""
    import torch.nn.functional as F
    from collections import defaultdict
    spec = TASK_SPECS[task_name]
    loader = (loaders['single_test'] if spec['kind'] == 'single'
              else loaders['pair_test'])
    wrapper.backbone.eval()
    wrapper.heads[task_name].eval()
    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for raw in loader:
            canvas, target = prepare_task_batch(task_name, raw, device=DEVICE)
            logits = wrapper.forward_task(canvas, task_name)
            from training import task_metric
            m = task_metric(task_name, logits, target)
            bs = canvas.size(0)
            for k, v in m.items():
                totals[k] += v * bs
            count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}
