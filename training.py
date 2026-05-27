"""
Training loops and optimizer construction for multi-task geodesic experiment.

Exports:
  task_loss, task_metric, eval_task
  compute_unified_loss
  estimate_perturb_lr_mult
  _make_optimizer
  run_spatial_probe
  train_single_task
  train_multitask
  train_sequential

Global training hyperparameters are exposed as module-level variables so the
CLI entry point (run_experiment.py) can override them before calling the
training functions.
"""

import os
import math
import time
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import (
    TASK_SPECS, TASK_WEIGHTS, TRAINED_TASKS, SEQUENTIAL_ORDER,
    NUM_STATE_VECTORS, CANVAS_W, MNIST_CROP_SIZE,
    prepare_task_batch, iter_task_batches,
    make_canvas_pair, make_canvas_single, crop_mnist,
    build_loaders, get_mnist_datasets,
)
from models.backbone import MultiTaskModel

# ============================================================
# Training hyperparameters (overridable by CLI)
# ============================================================

LR = 1e-3
EPOCHS = 50
EPOCHS_PER_SEQ_TASK = EPOCHS // len(SEQUENTIAL_ORDER)  # 10
TRANSFER_EPOCHS = 10
TRANSFER_LR = 1e-3

STATE_DIM = 12
DEPTH = 5
PERTURB_RANK = 2

PROBE_EPOCHS = set()           # epochs to run spatial probe
SEQ_PHASE_SNAPSHOTS = False    # G_seq: snapshot Grassmannian after each phase

# Unified geometric/topological loss weights (paper's three-pillar framework):
#   L_profile + L_rank target the GEOMETRIC pillar (Grassmannian)
#   L_topo targets the TOPOLOGICAL pillar (routing entropy)
PROFILE_WEIGHT = 0.0           # L_profile weight (shortcut penalty)
RANK_WEIGHT = 0.0              # L_rank weight (rank collapse penalty)
RANK_TARGET = 2.0              # minimum effective rank floor
TOPO_WEIGHT = 0.0              # L_topo weight (low-entropy routing penalty)

# Per-primitive LR multipliers
PERTURB_LR_MULT = 1.0          # LR multiplier for mixer perturbation params (U, V, eps)
TOPO_LR_MULT = 1.0             # LR multiplier for butterfly routing params (phi)

REG_PERTURB_L1 = 0.1           # L1 weight on mixer A_perturb (paper default)
SYM_REG = 0.0                  # ||A_sym||_F^2 weight

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = os.path.dirname(__file__)

MIXER_SKEW_ONLY = True
MIXER_USE_MATRIX_EXP = False

# ============================================================
# Loss and metric helpers
# ============================================================

def task_loss(task_name, logits, target):
    """Compute the primary task loss."""
    spec = TASK_SPECS[task_name]
    if spec['loss'] == 'ce':
        return F.cross_entropy(logits, target)
    elif spec['loss'] == 'mse':
        return F.mse_loss(logits, target)
    elif spec['loss'] == 'bce':
        return F.binary_cross_entropy_with_logits(logits, target)
    else:
        raise ValueError(f"Unknown loss: {spec['loss']}")


def task_metric(task_name, logits, target):
    """Return a dict with per-task primary metric value(s)."""
    spec = TASK_SPECS[task_name]
    if spec['metric'] == 'accuracy':
        if spec['loss'] == 'bce':
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct = (pred == target).float().mean().item()
        else:
            pred = logits.argmax(dim=-1)
            correct = (pred == target).float().mean().item()
        return {'accuracy': correct}
    elif spec['metric'] == 'mae':
        scale = spec.get('target_scale', 1.0)
        return {'mae': ((logits - target) * scale).abs().mean().item()}
    elif spec['metric'] == 'mse':
        return {'mse': ((logits - target) ** 2).mean().item()}
    else:
        raise ValueError(f"Unknown metric: {spec['metric']}")


def eval_task(model, task_name, loaders):
    """Evaluate model on the test set for one task."""
    spec = TASK_SPECS[task_name]
    loader = (loaders['single_test'] if spec['kind'] == 'single'
              else loaders['pair_test'])
    model.eval()
    totals = defaultdict(float)
    count = 0
    with torch.no_grad():
        for raw in loader:
            canvas, target = prepare_task_batch(task_name, raw, device=DEVICE)
            logits = model.forward_task(canvas, task_name)
            m = task_metric(task_name, logits, target)
            bs = canvas.size(0)
            for k, v in m.items():
                totals[k] += v * bs
            count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}


# ============================================================
# Unified geometric/topological loss
# ============================================================

def compute_unified_loss(model, task_name, layer_acts,
                          profile_weight=0.0, rank_weight=0.0, topo_weight=0.0,
                          rank_target=2.0):
    """Compute geometric and topological regularization terms.

    L_profile: penalizes high meaning fraction at early layers (shortcut).
    L_rank:    penalizes rank collapse (top PC dominance).
    L_topo:    penalizes low-entropy butterfly routing (encourages diverse routing).

    The three terms correspond to the paper's three-pillar diagnostic framework:
    geometric (L_profile + L_rank, Grassmannian) and topological (L_topo, routing).

    Returns dict with 'profile', 'rank', 'topo' as differentiable scalars.
    """
    device = layer_acts[0].device if layer_acts else next(model.parameters()).device
    L = len(layer_acts)
    zero = torch.zeros((), device=device)
    terms = {'profile': zero, 'rank': zero, 'topo': zero}

    if profile_weight > 0 and task_name in model.heads:
        W = model.heads[task_name].linear.weight        # (output_dim, d)
        W_n = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        profile_loss = torch.zeros((), device=device)
        for l, acts in enumerate(layer_acts):
            z = acts.mean(dim=1)
            z_n = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            cos2 = (z_n @ W_n.T).pow(2)
            mu_l = cos2.mean()
            w_l = 1.0 - float(l) / max(L - 1, 1)
            profile_loss = profile_loss + mu_l * w_l
        terms['profile'] = profile_weight * profile_loss

    if rank_weight > 0:
        rank_loss = torch.zeros((), device=device)
        target_ratio = 1.0 / max(rank_target, 1.0)
        for acts in layer_acts:
            z = acts.mean(dim=1).float()
            S = torch.linalg.svdvals(z)
            S2 = S.pow(2)
            top_ratio = S2[0] / (S2.sum() + 1e-12)
            violation = torch.clamp(top_ratio - target_ratio, min=0.0)
            rank_loss = rank_loss + violation.pow(2)
        terms['rank'] = rank_weight * rank_loss

    if topo_weight > 0 and getattr(model.backbone, 'butterflies', None):
        topo_loss = torch.zeros((), device=device)
        for bf in model.backbone.butterflies:
            if hasattr(bf, 'phi'):
                phi_abs = bf.phi.abs()
                phi_prob = (phi_abs + 1e-8) / (phi_abs + 1e-8).sum()
                entropy = -(phi_prob * torch.log(phi_prob + 1e-12)).sum()
                max_entropy = math.log(float(len(phi_abs)) + 1e-8)
                norm_entropy = entropy / max_entropy
                topo_loss = topo_loss + (1.0 - norm_entropy)
        terms['topo'] = topo_weight * topo_loss

    return terms


# ============================================================
# Gradient diagnostic
# ============================================================

def estimate_perturb_lr_mult(model, task_name, loaders):
    """One-batch gradient ratio measurement for perturbation LR calibration.

    Returns dict with 'task_grad_norm', 'rank_grad_norm', 'recommended_mult',
    and 'eps_values'.
    """
    model.train()
    raw = next(iter_task_batches(task_name, loaders))
    canvas, target = prepare_task_batch(task_name, raw, device=DEVICE)

    perturb_params = []
    for mixer in model.backbone.mixers:
        for attr in ('U', 'V', 'eps'):
            p = getattr(mixer, attr, None)
            if p is not None:
                perturb_params.append(p)

    if not perturb_params:
        return {'task_grad_norm': 0.0, 'rank_grad_norm': 0.0,
                'recommended_mult': 1.0, 'eps_values': []}

    logits, layer_acts = model.forward_task(canvas, task_name,
                                             capture_intermediates=True)
    loss_task = task_loss(task_name, logits, target)
    loss_task.backward(retain_graph=True)
    task_norms = [p.grad.norm().item() if p.grad is not None else 0.0
                  for p in perturb_params]
    task_grad_norm = float(np.mean(task_norms)) if task_norms else 0.0
    for p in perturb_params:
        if p.grad is not None:
            p.grad.zero_()

    geo = compute_unified_loss(model, task_name, layer_acts,
                                rank_weight=max(RANK_WEIGHT, 0.1),
                                rank_target=RANK_TARGET)
    geo['rank'].backward()
    rank_norms = [p.grad.norm().item() if p.grad is not None else 0.0
                  for p in perturb_params]
    rank_grad_norm = float(np.mean(rank_norms)) if rank_norms else 0.0
    for p in perturb_params:
        if p.grad is not None:
            p.grad.zero_()

    recommended = (task_grad_norm / rank_grad_norm
                   if rank_grad_norm > 1e-12 else 1.0)
    eps_values = [mixer.eps.item()
                  for mixer in model.backbone.mixers
                  if hasattr(mixer, 'eps')]

    return {
        'task_grad_norm': task_grad_norm,
        'rank_grad_norm': rank_grad_norm,
        'recommended_mult': recommended,
        'eps_values': eps_values,
    }


# ============================================================
# Optimizer
# ============================================================

def _make_optimizer(model, lr=None):
    """Build Adam with per-primitive-group learning rates.

    Three parameter groups matching the paper's three primitive types:
      1. task/backbone (lr):                  theta_head, pointwise, heads, preconditioner
      2. mixer perturbation (lr*PERTURB_LR_MULT): U, V, eps in RotationalProjectionMixer
      3. routing topology (lr*TOPO_LR_MULT):     phi in LearnedButterflyPermutation

    Group 2: the perturbation parameters sit at a near-zero attractor
    (dL/dU is proportional to eps, which is initialized to zero), so a higher
    LR multiplier helps them escape that basin when --rank-weight is active.
    Group 3: paired with --topo-weight, gives the routing primitive its own
    LR knob.
    """
    if lr is None:
        lr = LR
    if PERTURB_LR_MULT == 1.0 and TOPO_LR_MULT == 1.0:
        return optim.Adam(model.parameters(), lr=lr)

    perturb_ids, topo_ids = set(), set()
    for mixer in getattr(model.backbone, 'mixers', []):
        for name in ('U', 'V', 'eps'):
            p = getattr(mixer, name, None)
            if p is not None:
                perturb_ids.add(id(p))
    for bf in getattr(model.backbone, 'butterflies', []):
        p = getattr(bf, 'phi', None)
        if p is not None:
            topo_ids.add(id(p))

    perturb_params, topo_params, other_params = [], [], []
    for p in model.parameters():
        if id(p) in perturb_ids:
            perturb_params.append(p)
        elif id(p) in topo_ids:
            topo_params.append(p)
        else:
            other_params.append(p)

    groups = [{'params': other_params, 'lr': lr}]
    if perturb_params:
        groups.append({'params': perturb_params, 'lr': lr * PERTURB_LR_MULT})
    if topo_params:
        groups.append({'params': topo_params, 'lr': lr * TOPO_LR_MULT})
    return optim.Adam(groups)


# ============================================================
# Spatial probe
# ============================================================

def run_spatial_probe(model, loaders, task_name, epoch):
    """Frozen spatial-split linear probe for pairwise tasks (B_add, C_cmp).

    Splits 800 state vectors into left/right halves and fits LogisticRegression
    probes to predict individual digit classes from each half's pooled activations.

    Returns dict with per-half accuracy, cross-prediction (leakage), and
    effective rank.
    """
    if TASK_SPECS[task_name]['kind'] != 'pairwise':
        return {}

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  [probe] sklearn not available, skipping")
        return {}

    idx = torch.arange(NUM_STATE_VECTORS)
    left_mask = (idx % CANVAS_W) < MNIST_CROP_SIZE
    right_mask = ~left_mask

    model.eval()
    left_pool, right_pool, global_pool = [], [], []
    ya_all, yb_all = [], []

    with torch.no_grad():
        for raw_batch in loaders['pair_test']:
            img_a, lbl_a, img_b, lbl_b = raw_batch
            canvas = make_canvas_pair(
                crop_mnist(img_a.to(DEVICE)),
                crop_mnist(img_b.to(DEVICE))
            )
            feats = model.backbone(canvas)            # (B, 800, state_dim)
            left_pool.append(feats[:, left_mask, :].mean(1).cpu().numpy())
            right_pool.append(feats[:, right_mask, :].mean(1).cpu().numpy())
            global_pool.append(feats.mean(1).cpu().numpy())
            ya_all.append(lbl_a.numpy())
            yb_all.append(lbl_b.numpy())

    L = np.concatenate(left_pool)
    R = np.concatenate(right_pool)
    G = np.concatenate(global_pool)
    ya = np.concatenate(ya_all)
    yb = np.concatenate(yb_all)

    n = len(L)
    split = int(n * 0.8)

    def probe_acc(X, y):
        # solver='lbfgs' handles multinomial automatically in sklearn>=1.5;
        # the explicit multi_class='multinomial' arg was removed in sklearn 1.8.
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0,
                                 solver='lbfgs')
        clf.fit(X[:split], y[:split])
        return float(clf.score(X[split:], y[split:]))

    def eff_rank(X):
        S = np.linalg.svd(X - X.mean(0), compute_uv=False)
        S2 = S ** 2
        p = S2 / (S2.sum() + 1e-12)
        return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    results = {
        'epoch':           epoch,
        'left_to_left':    probe_acc(L, ya),
        'right_to_right':  probe_acc(R, yb),
        'left_to_right':   probe_acc(L, yb),
        'right_to_left':   probe_acc(R, ya),
        'global_to_left':  probe_acc(G, ya),
        'global_to_right': probe_acc(G, yb),
        'left_eff_rank':   eff_rank(L),
        'right_eff_rank':  eff_rank(R),
        'global_eff_rank': eff_rank(G),
    }

    print(f"  [spatial probe @ epoch {epoch}]")
    print(f"    left-half  -> left-digit:  {results['left_to_left']:.3f}   "
          f"rank={results['left_eff_rank']:.2f}")
    print(f"    right-half -> right-digit: {results['right_to_right']:.3f}   "
          f"rank={results['right_eff_rank']:.2f}")
    print(f"    global     -> left:  {results['global_to_left']:.3f}   "
          f"right: {results['global_to_right']:.3f}   "
          f"rank={results['global_eff_rank']:.2f}")
    print(f"    leakage    left->right: {results['left_to_right']:.3f}   "
          f"right->left: {results['right_to_left']:.3f}")
    return results


# ============================================================
# Gram warmup helper
# ============================================================

def _gram_warmup_if_needed(model, loaders, backbone_kind, gram_done=False, gram_delay=0, epoch=0):
    """Run gram_warmup at the right epoch. Returns updated gram_done flag."""
    if backbone_kind != 'gram_block':
        return True  # nothing to do
    if gram_done:
        return True
    if epoch < gram_delay:
        return False
    from models.backbone import GramBlockDiagonalMixer, GramRoutedPermutation

    def _compute_gram_sv_permutation(state_vectors):
        G = torch.einsum('bnd,bmd->nm', state_vectors, state_vectors)
        G = G / state_vectors.shape[0]
        _, eigvecs = torch.linalg.eigh(G)
        leading = eigvecs[:, -1]
        return leading.argsort()

    def _compute_gram_variable_blocks(pair_flat, num_blocks, min_block_size=3):
        pair_dim = pair_flat.shape[1]
        device = pair_flat.device
        if num_blocks <= 1:
            return torch.arange(pair_dim, device=device), [pair_dim]
        max_blocks = pair_dim // min_block_size
        num_blocks = min(num_blocks, max_blocks)
        G = pair_flat.T @ pair_flat / pair_flat.shape[0]
        feature_var = G.diag()
        perm = feature_var.argsort(descending=True)
        sorted_var = feature_var[perm]
        cum_var = sorted_var.cumsum(0)
        total_var = cum_var[-1].item()
        if total_var < 1e-12:
            from models.backbone import _equal_block_sizes
            return perm, _equal_block_sizes(pair_dim, num_blocks)
        target_per_block = total_var / num_blocks
        block_sizes = []
        prev = 0
        for b in range(num_blocks - 1):
            target = target_per_block * (b + 1)
            mask = cum_var >= target
            candidates = mask.nonzero(as_tuple=True)[0]
            boundary = candidates[0].item() + 1 if len(candidates) > 0 else pair_dim
            boundary = max(boundary, prev + min_block_size)
            remaining_blocks = num_blocks - b - 1
            max_boundary = pair_dim - remaining_blocks * min_block_size
            boundary = min(boundary, max_boundary)
            block_sizes.append(boundary - prev)
            prev = boundary
        block_sizes.append(pair_dim - prev)
        return perm, block_sizes

    backbone = model.backbone
    backbone.eval()
    with torch.no_grad():
        raw = next(iter(loaders['single_train']))
        canvas = make_canvas_single(crop_mnist(raw[0].to(DEVICE)))
        sv = backbone.encoder.encode_mnist(canvas)
        if backbone.preconditioner is not None:
            sv = backbone.preconditioner(sv)
        B_w, N_w, d_w = sv.shape
        padded = sv.new_zeros(B_w, N_w, backbone.state_dim)
        padded[:, :, :d_w] = sv
        sv = padded
        sv_perm = _compute_gram_sv_permutation(sv)
        for prim in backbone.primitive_factorization.primitives:
            if isinstance(prim, GramRoutedPermutation):
                prim.set_permutation(sv_perm)
        sv_routed = sv[:, sv_perm, :]
        s_i = sv_routed[:, 0::2, :]
        s_j = sv_routed[:, 1::2, :]
        pair_vectors = torch.cat([s_i, s_j], dim=2)
        pair_flat = pair_vectors.reshape(-1, 2 * backbone.state_dim)
        num_blocks = backbone.num_blocks
        feat_perm, block_sizes = _compute_gram_variable_blocks(pair_flat, num_blocks)
        for mixer in backbone.mixers:
            if isinstance(mixer, GramBlockDiagonalMixer):
                mixer.set_feature_permutation(feat_perm)
                mixer.rebuild_blocks(block_sizes)
    backbone.train()
    return True


# ============================================================
# Single-task training
# ============================================================

def train_single_task(task_name, seed, loaders, epochs=None, verbose=True,
                      backbone_kind='rotational', state_dim=None,
                      num_blocks=1, gram_delay=0):
    """Train a fresh model on a single task."""
    if epochs is None:
        epochs = EPOCHS
    if state_dim is None:
        state_dim = STATE_DIM

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MultiTaskModel(
        task_names=[task_name], state_dim=state_dim, backbone_kind=backbone_kind,
        num_blocks=num_blocks,
        mixer_skew_only=MIXER_SKEW_ONLY,
        mixer_use_matrix_exp=MIXER_USE_MATRIX_EXP,
    ).to(DEVICE)

    gram_done = _gram_warmup_if_needed(
        model, loaders, backbone_kind, gram_done=False,
        gram_delay=gram_delay, epoch=0
    ) if gram_delay <= 0 else False

    optimizer = _make_optimizer(model)

    if (RANK_WEIGHT > 0 or PROFILE_WEIGHT > 0) and verbose:
        diag = estimate_perturb_lr_mult(model, task_name, loaders)
        print(f"    [grad-ratio] task={diag['task_grad_norm']:.4e}  "
              f"rank={diag['rank_grad_norm']:.4e}  "
              f"recommended_mult={diag['recommended_mult']:.1f}x  "
              f"current_mult={PERTURB_LR_MULT:.1f}x")
        print(f"    [init eps]  {[f'{v:.4f}' for v in diag['eps_values']]}")

    history = {
        'train_loss': [], 'test_metric': [], 'epoch_seconds': [],
        'perturb_eps': [], 'perturb_U_norm': [], 'perturb_V_norm': [],
        'perturb_A_norm': [], 'perturb_grad_norm': [],
    }

    ckpt_dir = os.path.join(SAVE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        if not gram_done and epoch == gram_delay:
            if verbose:
                print(f"    -- Gram warmup at epoch {epoch} "
                      f"(delayed {gram_delay} epochs) --")
            gram_done = _gram_warmup_if_needed(
                model, loaders, backbone_kind, gram_done=False,
                gram_delay=0, epoch=0
            )
            optimizer = _make_optimizer(model)

        epoch_t0 = time.time()
        model.train()
        losses = []
        unified = {'profile': [], 'rank': [], 'topo': []}
        need_intermediates = PROFILE_WEIGHT > 0 or RANK_WEIGHT > 0
        _ps = {'eps': 0.0, 'U': 0.0, 'V': 0.0, 'A': 0.0, 'grad': 0.0}

        for raw in iter_task_batches(task_name, loaders):
            canvas, target = prepare_task_batch(task_name, raw, device=DEVICE)
            optimizer.zero_grad()
            if need_intermediates:
                logits, layer_acts = model.forward_task(
                    canvas, task_name, capture_intermediates=True)
            else:
                logits = model.forward_task(canvas, task_name)
                layer_acts = []
            total = task_loss(task_name, logits, target)
            if torch.isnan(total):
                if verbose:
                    print(f"    ** NaN at epoch {epoch+1} **")
                break
            if hasattr(model.backbone, 'mixers'):
                for mixer in model.backbone.mixers:
                    if hasattr(mixer, 'regularization_terms'):
                        reg = mixer.regularization_terms()
                        if REG_PERTURB_L1 > 0:
                            total = total + REG_PERTURB_L1 * reg['a_perturb_l1']
                        if SYM_REG > 0:
                            total = total + SYM_REG * reg['a_sym_frob']
            if layer_acts or TOPO_WEIGHT > 0:
                geo = compute_unified_loss(
                    model, task_name, layer_acts,
                    profile_weight=PROFILE_WEIGHT,
                    rank_weight=RANK_WEIGHT,
                    topo_weight=TOPO_WEIGHT,
                    rank_target=RANK_TARGET,
                )
                total = total + geo['profile'] + geo['rank'] + geo['topo']
                unified['profile'].append(geo['profile'].item())
                unified['rank'].append(geo['rank'].item())
                unified['topo'].append(geo['topo'].item())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if not losses:  # first batch of this epoch: capture perturb state
                eps_vals, U_norms, V_norms, A_norms, g_norms = [], [], [], [], []
                for mx in getattr(model.backbone, 'mixers', []):
                    if hasattr(mx, 'eps'):
                        eps_vals.append(abs(mx.eps.item()))
                    if hasattr(mx, 'U'):
                        U_norms.append(mx.U.data.norm().item())
                        g_norms.append(mx.U.grad.norm().item()
                                       if mx.U.grad is not None else 0.0)
                    if hasattr(mx, 'V'):
                        V_norms.append(mx.V.data.norm().item())
                        g_norms.append(mx.V.grad.norm().item()
                                       if mx.V.grad is not None else 0.0)
                    if hasattr(mx, '_build_A_perturb'):
                        with torch.no_grad():
                            A_norms.append(mx._build_A_perturb().norm().item())
                _ps = {
                    'eps':  float(np.mean(eps_vals))  if eps_vals  else 0.0,
                    'U':    float(np.mean(U_norms))   if U_norms   else 0.0,
                    'V':    float(np.mean(V_norms))   if V_norms   else 0.0,
                    'A':    float(np.mean(A_norms))   if A_norms   else 0.0,
                    'grad': float(np.mean(g_norms))   if g_norms   else 0.0,
                }

            optimizer.step()
            losses.append(total.item())

        train_loss = float(np.mean(losses)) if losses else float('nan')
        test_m = eval_task(model, task_name, loaders)
        epoch_sec = time.time() - epoch_t0
        history['train_loss'].append(train_loss)
        history['test_metric'].append(test_m)
        history['epoch_seconds'].append(epoch_sec)
        history['perturb_eps'].append(_ps['eps'])
        history['perturb_U_norm'].append(_ps['U'])
        history['perturb_V_norm'].append(_ps['V'])
        history['perturb_A_norm'].append(_ps['A'])
        history['perturb_grad_norm'].append(_ps['grad'])

        if verbose:
            geo_str = ''
            if unified['profile'] or unified['rank'] or unified['topo']:
                p = float(np.mean(unified['profile'])) if unified['profile'] else 0.0
                r = float(np.mean(unified['rank']))     if unified['rank']    else 0.0
                t = float(np.mean(unified['topo']))     if unified['topo']    else 0.0
                geo_str = f'  geo=(p={p:.4f} r={r:.4f} t={t:.4f})'
            perturb_str = (f'  perturb=(eps={_ps["eps"]:.3f} '
                           f'|U|={_ps["U"]:.3f} |V|={_ps["V"]:.3f} '
                           f'|A|={_ps["A"]:.4f} grad={_ps["grad"]:.3e})')
            print(f"    epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, "
                  f"test={test_m}{geo_str}{perturb_str}  [{epoch_sec:.0f}s]")

        ckpt_path = os.path.join(
            ckpt_dir, f'single_{task_name}_{backbone_kind}_s{seed}_e{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_metric': test_m,
            'history': history,
        }, ckpt_path)

        if (epoch + 1) in PROBE_EPOCHS:
            probe = run_spatial_probe(model, loaders, task_name, epoch + 1)
            history.setdefault('spatial_probes', []).append(probe)

        spec = TASK_SPECS[task_name]
        if spec['metric'] == 'accuracy':
            primary = list(test_m.values())[0]
            if primary >= 0.90:
                if verbose:
                    print(f"    early exit at epoch {epoch+1}: "
                          f"primary metric {primary:.4f} >= 0.90")
                break

    return model, history


# ============================================================
# Multi-task training (Condition F)
# ============================================================

def train_multitask(task_names, seed, loaders, epochs=None,
                    task_weights=None, verbose=True,
                    backbone_kind='rotational', state_dim=None,
                    num_blocks=1, gram_delay=0,
                    reg_perturb_l1=0.0):
    """Condition F: joint training with summed losses."""
    if epochs is None:
        epochs = EPOCHS
    if state_dim is None:
        state_dim = STATE_DIM
    if task_weights is None:
        task_weights = TASK_WEIGHTS

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MultiTaskModel(
        task_names=task_names, state_dim=state_dim, backbone_kind=backbone_kind,
        num_blocks=num_blocks,
        mixer_skew_only=MIXER_SKEW_ONLY,
        mixer_use_matrix_exp=MIXER_USE_MATRIX_EXP,
    ).to(DEVICE)

    gram_done = _gram_warmup_if_needed(
        model, loaders, backbone_kind, gram_done=False,
        gram_delay=gram_delay, epoch=0
    ) if gram_delay <= 0 else False

    optimizer = _make_optimizer(model)

    history = {'train_loss': [], 'test_metric': {t: [] for t in task_names},
               'epoch_seconds': []}

    ckpt_dir = os.path.join(SAVE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        if not gram_done and epoch == gram_delay:
            if verbose:
                print(f"    -- Gram warmup at epoch {epoch} "
                      f"(delayed {gram_delay} epochs) --")
            gram_done = _gram_warmup_if_needed(
                model, loaders, backbone_kind, gram_done=False,
                gram_delay=0, epoch=0
            )
            optimizer = _make_optimizer(model)

        epoch_t0 = time.time()
        model.train()
        task_iters = {t: iter_task_batches(t, loaders) for t in task_names}
        step_losses = []
        while True:
            try:
                raws = {t: next(task_iters[t]) for t in task_names}
            except StopIteration:
                break

            optimizer.zero_grad()
            total = 0.0
            need_intermediates = PROFILE_WEIGHT > 0 or RANK_WEIGHT > 0
            for t in task_names:
                canvas, target = prepare_task_batch(t, raws[t], device=DEVICE)
                if need_intermediates:
                    logits, layer_acts_t = model.forward_task(
                        canvas, t, capture_intermediates=True)
                else:
                    logits = model.forward_task(canvas, t)
                    layer_acts_t = []
                loss_t = task_loss(t, logits, target)
                total = total + task_weights.get(t, 1.0) * loss_t
                if layer_acts_t or TOPO_WEIGHT > 0:
                    geo = compute_unified_loss(
                        model, t, layer_acts_t,
                        profile_weight=PROFILE_WEIGHT,
                        rank_weight=RANK_WEIGHT,
                        topo_weight=TOPO_WEIGHT,
                        rank_target=RANK_TARGET,
                    )
                    total = total + geo['profile'] + geo['rank'] + geo['topo']

            for mixer in model.backbone.mixers:
                if hasattr(mixer, 'regularization_terms'):
                    reg = mixer.regularization_terms()
                    if reg_perturb_l1 > 0:
                        total = total + reg_perturb_l1 * reg['a_perturb_l1']
                    if SYM_REG > 0:
                        total = total + SYM_REG * reg.get('a_sym_frob', 0.0)

            if torch.is_tensor(total) and torch.isnan(total):
                if verbose:
                    print(f"    ** NaN at epoch {epoch+1} **")
                break
            if torch.is_tensor(total):
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step_losses.append(total.item())

        train_loss = float(np.mean(step_losses)) if step_losses else float('nan')
        history['train_loss'].append(train_loss)

        for t in task_names:
            history['test_metric'][t].append(eval_task(model, t, loaders))

        epoch_sec = time.time() - epoch_t0
        history['epoch_seconds'].append(epoch_sec)

        if verbose:
            summary = ', '.join(
                f"{t}={list(history['test_metric'][t][-1].values())[0]:.3f}"
                for t in task_names
            )
            print(f"    epoch {epoch+1}/{epochs}: loss={train_loss:.4f} | "
                  f"{summary}  [{epoch_sec:.0f}s]")

        cond_tag = backbone_kind
        if backbone_kind == 'gram_block':
            cond_tag = f'gram_block_{num_blocks}'
        ckpt_path = os.path.join(
            ckpt_dir, f'multi_{cond_tag}_s{seed}_e{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_metric': {t: history['test_metric'][t][-1] for t in task_names},
            'history': history,
        }, ckpt_path)

    return model, history


# ============================================================
# Sequential training (Condition G)
# ============================================================

def train_sequential(task_order, seed, loaders, epochs_per_task=None,
                     verbose=True, backbone_kind='rotational', state_dim=None,
                     phase_snapshots=False):
    """Condition G: cumulative curriculum -- each phase ADDS a new task.

    Phase 1: train on task_order[0]
    Phase 2: train on task_order[0:2]
    ...

    Prevents catastrophic forgetting by maintaining all active task heads
    and the backbone jointly during each phase.
    """
    if epochs_per_task is None:
        epochs_per_task = EPOCHS_PER_SEQ_TASK
    if state_dim is None:
        state_dim = STATE_DIM

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MultiTaskModel(
        task_names=task_order, backbone_kind=backbone_kind, state_dim=state_dim,
        mixer_skew_only=MIXER_SKEW_ONLY,
        mixer_use_matrix_exp=MIXER_USE_MATRIX_EXP,
    ).to(DEVICE)

    history = {
        'train_loss': [],
        'test_metric': {t: [] for t in task_order},
        'phase': [],
        'phase_snapshots': [],
    }

    for phase_idx in range(len(task_order)):
        active_tasks = task_order[:phase_idx + 1]
        new_task = task_order[phase_idx]

        trainable_params = list(model.backbone.parameters())
        for t in active_tasks:
            trainable_params += list(model.heads[t].parameters())
        optimizer = optim.Adam(trainable_params, lr=LR)

        if verbose:
            print(f"    -- phase {phase_idx+1}/{len(task_order)}: "
                  f"add {new_task} (active: {active_tasks}) --")

        for ep in range(epochs_per_task):
            model.train()
            losses = []

            iters = {t: iter(iter_task_batches(t, loaders))
                     for t in active_tasks}
            exhausted = set()
            while len(exhausted) < len(active_tasks):
                batch_loss = torch.tensor(0.0, device=DEVICE)
                for t in active_tasks:
                    if t in exhausted:
                        continue
                    try:
                        raw = next(iters[t])
                    except StopIteration:
                        exhausted.add(t)
                        continue
                    canvas, target = prepare_task_batch(t, raw, device=DEVICE)
                    logits = model.forward_task(canvas, t)
                    w = TASK_WEIGHTS.get(t, 1.0)
                    batch_loss = batch_loss + w * task_loss(t, logits, target)

                if batch_loss.item() == 0.0:
                    break
                optimizer.zero_grad()
                total = batch_loss
                if hasattr(model.backbone, 'mixers'):
                    for mixer in model.backbone.mixers:
                        if hasattr(mixer, 'regularization_terms'):
                            reg = mixer.regularization_terms()
                            if REG_PERTURB_L1 > 0:
                                total = total + REG_PERTURB_L1 * reg['a_perturb_l1']
                            if SYM_REG > 0:
                                total = total + SYM_REG * reg.get('a_sym_frob', 0.0)
                if torch.isnan(total):
                    if verbose:
                        print("    ** NaN **")
                    break
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(total.item())

            train_loss = float(np.mean(losses)) if losses else float('nan')
            history['train_loss'].append(train_loss)
            history['phase'].append(new_task)
            for t in task_order:
                history['test_metric'][t].append(eval_task(model, t, loaders))

            if verbose:
                metrics = {t: eval_task(model, t, loaders) for t in active_tasks}
                metric_str = ', '.join(f'{t}={list(m.values())[0]:.3f}'
                                       for t, m in metrics.items())
                print(f"      epoch {ep+1}/{epochs_per_task}: "
                      f"loss={train_loss:.4f} | {metric_str}")

        if phase_snapshots:
            if verbose:
                print(f"    [phase snapshot {phase_idx+1}: active={active_tasks}]")
            from analysis import compute_grassmannian_suite
            heads_so_far = {t: model.heads[t] for t in active_tasks}
            snap = compute_grassmannian_suite(
                model, heads_so_far, loaders, num_samples=512
            )
            snap['active_tasks'] = list(active_tasks)
            snap['phase_idx'] = phase_idx
            snap['new_task'] = new_task
            history['phase_snapshots'].append(snap)
            if verbose:
                print(f"      reservoir_rank={snap['reservoir_effective_rank']:.3f}")

    return model, history
