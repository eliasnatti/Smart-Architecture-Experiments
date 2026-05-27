"""
run_experiment.py -- CLI entry point for the multi-task MNIST geodesic experiment.

Usage examples:

    # Full experiment (all 7 conditions, 3 seeds each):
    python run_experiment.py

    # Single condition:
    python run_experiment.py --conditions B_add --seeds 0 1 2

    # 200-epoch spatial probe run (B_add only):
    python run_experiment.py --conditions B_add --seeds 0 \\
        --epochs 200 --probe-epochs 50 100 150 200 \\
        --profile-weight 0.5 --rank-weight 0.1 --rank-target 4.0 \\
        --perturb-lr-mult 10.0 --run-id long_probe

    # G_seq with per-phase reservoir snapshots:
    python run_experiment.py --conditions G_seq --seeds 0 1 2 \\
        --seq-phase-snapshots --seq-epochs-per-phase 50 --run-id gseq_phases

    # Regenerate figures from saved results:
    python plots.py results/exp7_results_20260515_212500.json
"""

import os
import sys
import time
import json
import math
import argparse

import numpy as np
import torch

# ============================================================
# Condition tables
# ============================================================

SINGLE_TASK_CONDITIONS = ['A_cls', 'B_add', 'C_cmp', 'D_spa', 'E_oe']
CONDITION_TO_TASK = {
    'A_cls': 'classification',
    'B_add': 'addition',
    'C_cmp': 'comparison',
    'D_spa': 'spatial',
    'E_oe':  'odd_even',
}
CONDITIONS = SINGLE_TASK_CONDITIONS + ['F_multi', 'G_seq']

# Architecture variant per condition (default: 'rotational')
BACKBONE_KIND = {}
ROTATIONAL_STATE_DIM = {}
GRAM_NUM_BLOCKS = {}
GRAM_DELAY = {}

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Per-condition runner
# ============================================================

def run_condition(cond_name, seed, loaders):
    """Dispatch a single (condition, seed) training run."""
    import training as T
    from training import eval_task
    from analysis import compute_grassmannian_suite, transfer_to_task6
    from data import TRAINED_TASKS

    print(f"\n{'='*60}\nCondition {cond_name} | seed {seed}\n{'='*60}")

    kind = BACKBONE_KIND.get(cond_name, 'rotational')
    sd = ROTATIONAL_STATE_DIM.get(cond_name, T.STATE_DIM)
    nb = GRAM_NUM_BLOCKS.get(cond_name, 1)
    gd = GRAM_DELAY.get(cond_name, 0)

    if kind == 'rotational':
        reg_str = f', reg_perturb_l1={T.REG_PERTURB_L1}' if T.REG_PERTURB_L1 > 0 else ''
        if T.MIXER_SKEW_ONLY:
            mixer_str = 'skew_only'
        elif T.MIXER_USE_MATRIX_EXP:
            mixer_str = 'full_matrix_exp'
        else:
            mixer_str = 'full_pade'
        print(f"  Backbone: H_rotational (state_dim={sd}, mixer={mixer_str}{reg_str})")
    elif kind == 'gram_block':
        print(f"  Backbone: gram_block (state_dim={sd}, num_blocks={nb}, "
              f"gram_delay={gd})")

    if cond_name in SINGLE_TASK_CONDITIONS:
        task = CONDITION_TO_TASK[cond_name]
        model, history = T.train_single_task(
            task, seed, loaders, backbone_kind=kind, state_dim=sd,
            epochs=T.EPOCHS,
        )
    elif cond_name == 'F_multi':
        from data import TRAINED_TASKS as _TT
        model, history = T.train_multitask(
            _TT, seed, loaders, backbone_kind=kind, state_dim=sd,
            reg_perturb_l1=T.REG_PERTURB_L1,
        )
    elif kind == 'gram_block':
        from data import TRAINED_TASKS as _TT
        model, history = T.train_multitask(
            _TT, seed, loaders, backbone_kind=kind, state_dim=sd,
            num_blocks=nb, gram_delay=gd,
        )
    elif cond_name == 'G_seq':
        from data import SEQUENTIAL_ORDER
        model, history = T.train_sequential(
            SEQUENTIAL_ORDER, seed, loaders, backbone_kind=kind, state_dim=sd,
            epochs_per_task=T.EPOCHS_PER_SEQ_TASK,
            phase_snapshots=T.SEQ_PHASE_SNAPSHOTS,
        )
    else:
        raise ValueError(f"Unknown condition: {cond_name}")

    # Final per-task evaluation
    print("  Evaluating final test metrics...")
    final_test = {}
    for t in TRAINED_TASKS:
        if t in model.heads:
            final_test[t] = eval_task(model, t, loaders)
    print(f"  Final test: {final_test}")

    # Grassmannian measurements
    print("  Computing Grassmannian measurements...")
    heads_by_task = {t: model.heads[t] for t in model.heads}
    grassmann = compute_grassmannian_suite(
        model, heads_by_task, loaders, num_samples=512
    )
    eig_full = [f"{s['effective_rank']:.1f}"
                for s in grassmann['mixer_eigenvalue_spectrum']]
    eig_sym = [f"{s.get('sym_effective_rank', 0):.1f}"
               for s in grassmann['mixer_eigenvalue_spectrum']]
    eig_skew = [f"{s.get('skew_effective_rank', 0):.1f}"
                for s in grassmann['mixer_eigenvalue_spectrum']]
    print(f"  Grassmannian: reservoir_rank={grassmann['reservoir_effective_rank']:.2f}")
    print(f"    mixer eig ranks (full): {eig_full}")
    print(f"    mixer eig ranks (sym/scaling): {eig_sym}")
    print(f"    mixer eig ranks (skew/rotation): {eig_skew}")

    if model.backbone.butterflies:
        bf_stats = [bf.routing_stats() for bf in model.backbone.butterflies]
        bf_active = [f"{s['active']:.0%}" for s in bf_stats]
        bf_phi = [f"{s['mean_phi']:.2f}" for s in bf_stats]
        print(f"    butterfly active (|phi|>0.1): {bf_active}")
        print(f"    butterfly mean |phi|: {bf_phi}")

    # Transfer to Task 6
    print("  Training transfer head (Task 6)...")
    transfer = transfer_to_task6(model.backbone, loaders, seed=seed)
    print(f"  Transfer acc: {transfer}")

    return {
        'history': history,
        'final_test': final_test,
        'grassmann': grassmann,
        'transfer': transfer,
    }


# ============================================================
# Result serialization
# ============================================================

def _save_results(all_results, path):
    def _scrub(v):
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_scrub(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        return v

    data = _scrub(all_results)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='ascii') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


# ============================================================
# Main experiment driver
# ============================================================

def run_experiment(conditions=None, seeds=None):
    import training as T
    from data import get_mnist_datasets, build_loaders
    from plots import _aggregate_across_seeds, _make_plots

    if conditions is None:
        conditions = CONDITIONS
    if seeds is None:
        seeds = list(range(3))

    print(f"Device: {T.DEVICE}")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    print(f"State dim: {T.STATE_DIM}, depth: {T.DEPTH}, epochs: {T.EPOCHS}")

    t0 = time.time()
    train_ds, test_ds = get_mnist_datasets()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    rrun_id = getattr(T, 'RUN_ID', '')
    suffix = f'_{rrun_id}' if rrun_id else ''
    results_dir = os.path.join(SAVE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'exp7_results_{timestamp}{suffix}.json')
    print(f"Results will be saved to: {results_path}")

    all_results = {}
    for cond in conditions:
        for seed in seeds:
            loaders = build_loaders(train_ds, test_ds, seed=seed)
            key = f"{cond}_s{seed}"
            all_results[key] = run_condition(cond, seed, loaders)
            _save_results(all_results, path=results_path)

    aggregated = _aggregate_across_seeds(all_results, conditions, seeds)
    # When --run-id is set, namespace plots as exp7_<run_id>_*.png so smoke
    # tests and exploratory runs don't overwrite the canonical paper figures.
    plot_prefix = f'exp7_{rrun_id}_' if rrun_id else 'exp7_'
    _make_plots(aggregated, save_dir=results_dir, prefix=plot_prefix,
                all_results=all_results, seeds=seeds)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import training as T

    parser = argparse.ArgumentParser(
        description='Multi-task MNIST geodesic coverage experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sel = parser.add_argument_group('experiment selection')
    sel.add_argument('--conditions', nargs='*', default=None,
                     help='Subset of conditions to run. Default: all 7 '
                          '(A_cls, B_add, C_cmp, D_spa, E_oe, F_multi, G_seq).')
    sel.add_argument('--seeds', type=int, nargs='*', default=None,
                     help='Seeds to use. Default: [0, 1, 2].')
    sel.add_argument('--epochs', type=int, default=None, metavar='N',
                     help='Training epochs per condition. Default: 50.')
    sel.add_argument('--run-id', type=str, default='', metavar='ID',
                     help='Label appended to the results filename.')

    probes = parser.add_argument_group('probes and analysis')
    probes.add_argument('--probe-epochs', type=int, nargs='*', default=[],
                        metavar='E',
                        help='Epochs at which to run the frozen spatial probe '
                             '(pairwise tasks only). E.g. --probe-epochs 50 100 150 200.')
    probes.add_argument('--seq-phase-snapshots', action='store_true', default=False,
                        help='G_seq: take a full Grassmannian snapshot after each '
                             'curriculum phase. Enables rank-trajectory analysis.')
    probes.add_argument('--seq-epochs-per-phase', type=int, default=None, metavar='N',
                        help='G_seq: epochs per curriculum phase. '
                             'Default: EPOCHS // 5.')

    mixer = parser.add_argument_group('mixer variants (paper ablations)')
    mixer.add_argument('--reg-perturb-l1', type=float, default=0.1, metavar='W',
                       help='L1 penalty on mixer A_perturb (UV^T - VU^T). '
                            'Drives unused perturbation layers to zero, revealing '
                            'per-task complexity requirements. Default 0.1 matches '
                            'the paper runs. Pass 0.0 to disable for ablation.')
    mixer.add_argument('--full-exp-a', action='store_true',
                       help='Use full exp(A) = exp(A_sym + A_skew) instead of '
                            'the default skew-only mixer. Adds magnitude scaling '
                            'alongside rotation.')
    mixer.add_argument('--full-matrix-exp', action='store_true',
                       help='Use exact torch.linalg.matrix_exp instead of the '
                            'Pade [1,1] approximation. ~3-5x slower; only meaningful '
                            'when combined with --full-exp-a.')
    mixer.add_argument('--sym-reg', type=float, default=0.0, metavar='W',
                       help='Weight on ||A_sym||_F^2 penalty. Used in the '
                            'pade_symreg / matexp_symreg replications.')

    unified = parser.add_argument_group(
        'unified loss penalties (three-pillar diagnostic framework)'
    )
    unified.add_argument('--profile-weight', type=float, default=0.0, metavar='W',
                         help='[geometric] Weight on L_profile. Penalizes high meaning '
                              'fraction at early layers (shortcut prevention).')
    unified.add_argument('--rank-weight', type=float, default=0.0, metavar='W',
                         help='[geometric] Weight on L_rank. Pushes reservoir effective '
                              'rank toward --rank-target (rank-collapse prevention).')
    unified.add_argument('--rank-target', type=float, default=2.0, metavar='R',
                         help='Target effective rank floor for L_rank. Default: 2.0.')
    unified.add_argument('--topo-weight', type=float, default=0.0, metavar='W',
                         help='[topological] Weight on L_topo. Penalizes low-entropy '
                              'butterfly routing (encourages diverse permutation paths). '
                              'Never set non-zero in the paper runs; included for the '
                              'three-pillar framework.')
    unified.add_argument('--perturb-lr-mult', type=float, default=1.0, metavar='M',
                         help='LR multiplier for mixer perturbation params (U, V, eps). '
                              'These sit at a near-zero attractor; a higher multiplier '
                              'helps escape the basin. Suggested: 5-20 with --rank-weight.')
    unified.add_argument('--topo-lr-mult', type=float, default=1.0, metavar='M',
                         help='LR multiplier for butterfly routing params (phi). '
                              'Paired with --topo-weight.')

    gram = parser.add_argument_group('GramBlock variant (alternative architecture)')
    gram.add_argument('--gram-blocks', type=int, nargs='+', default=[],
                      metavar='K',
                      help='Add Gram-routed block-diagonal conditions. Accepts one or '
                           'more block counts, e.g. --gram-blocks 3 6. See README note '
                           'on GramBlock at low state_dim.')
    gram.add_argument('--gram-delay', type=int, default=0, metavar='N',
                      help='Epochs before Gram warmup. 0 = warmup before training.')

    args = parser.parse_args()

    conditions = args.conditions
    seeds = args.seeds

    # Apply CLI overrides to training module globals
    T.RUN_ID = args.run_id
    T.REG_PERTURB_L1 = args.reg_perturb_l1
    T.SYM_REG = args.sym_reg

    if args.full_exp_a:
        T.MIXER_SKEW_ONLY = False
        T.MIXER_USE_MATRIX_EXP = args.full_matrix_exp

    T.PROFILE_WEIGHT = args.profile_weight
    T.RANK_WEIGHT = args.rank_weight
    T.RANK_TARGET = args.rank_target
    T.TOPO_WEIGHT = args.topo_weight
    T.PERTURB_LR_MULT = args.perturb_lr_mult
    T.TOPO_LR_MULT = args.topo_lr_mult

    if args.epochs is not None:
        T.EPOCHS = args.epochs
        print(f"Epochs override: {T.EPOCHS}")

    T.PROBE_EPOCHS = set(args.probe_epochs) if args.probe_epochs else set()
    T.SEQ_PHASE_SNAPSHOTS = args.seq_phase_snapshots

    if args.seq_epochs_per_phase is not None:
        T.EPOCHS_PER_SEQ_TASK = args.seq_epochs_per_phase
        print(f"G_seq epochs per phase: {T.EPOCHS_PER_SEQ_TASK}")

    pair_dim = 2 * T.STATE_DIM
    for K in args.gram_blocks:
        gb_name = f'GB{K}_multi'
        BACKBONE_KIND[gb_name] = 'gram_block'
        ROTATIONAL_STATE_DIM[gb_name] = T.STATE_DIM
        GRAM_NUM_BLOCKS[gb_name] = K
        GRAM_DELAY[gb_name] = args.gram_delay
        if conditions is None:
            conditions = [gb_name]
        elif gb_name not in conditions:
            conditions.append(gb_name)
        print(f"Added condition {gb_name}: {K} blocks "
              f"within pair_dim={pair_dim}, gram_delay={args.gram_delay}")

    run_experiment(conditions=conditions, seeds=seeds)
