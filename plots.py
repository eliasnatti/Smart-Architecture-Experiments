"""
Figure generation for the multi-task MNIST geodesic experiment.

All plot_* functions accept an aggregated results dict (keyed by condition name)
and write PNG files to a specified path.

CLI usage:
    python plots.py results/exp7_results_20260515_212500.json

This regenerates all figures from saved results JSON.
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import TRAINED_TASKS, SEQUENTIAL_ORDER


# ============================================================
# Individual plot functions
# ============================================================

def plot_latent_occupancy(results, save_path):
    """Layer-wise off-diagonal energy fraction per condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, r in results.items():
        occ = np.array(r['grassmann']['layer_occupancy'])
        ax.plot(range(1, len(occ) + 1), occ, marker='o', label=cond)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Off-diagonal energy fraction')
    ax.set_title('Latent Space Occupancy vs Depth')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_meaning_heatmap(results, save_path):
    """Conditions x Tasks heatmap of mean per-layer meaning fraction."""
    conds = list(results.keys())
    tasks = TRAINED_TASKS
    mat = np.zeros((len(conds), len(tasks)))
    for i, c in enumerate(conds):
        mpt = results[c]['grassmann']['meaning_per_task']
        for j, t in enumerate(tasks):
            vals = mpt.get(t, [])
            mat[i, j] = float(np.mean(vals)) if vals else float('nan')

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.set_title('Meaning fraction mu_ell (avg over layers)')
    for i in range(len(conds)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                    color='white', fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_meaning_per_layer_heatmap(results, save_path):
    """One subplot per condition: tasks x layers heatmap of mu_ell."""
    conds = list(results.keys())
    tasks = TRAINED_TASKS
    n_cond = len(conds)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4), squeeze=False)
    for ci, c in enumerate(conds):
        mpt = results[c]['grassmann']['meaning_per_task']
        n_layers = 0
        for t in tasks:
            vals = mpt.get(t, [])
            if vals:
                n_layers = max(n_layers, len(vals))
        if n_layers == 0:
            continue
        mat = np.full((len(tasks), n_layers), np.nan)
        for ti, t in enumerate(tasks):
            vals = mpt.get(t, [])
            for li, v in enumerate(vals):
                mat[ti, li] = float(v)
        ax = axes[0, ci]
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{l+1}" for l in range(n_layers)])
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_title(f"{c}: mu_ell per (task, layer)")
        for ti in range(len(tasks)):
            for li in range(n_layers):
                if not np.isnan(mat[ti, li]):
                    ax.text(li, ti, f"{mat[ti, li]:.2f}",
                            ha='center', va='center', color='white', fontsize=7)
            row = mat[ti]
            if not np.all(np.isnan(row)):
                peak = int(np.nanargmax(row))
                ax.add_patch(plt.Rectangle((peak - 0.5, ti - 0.5), 1, 1,
                                           fill=False, edgecolor='red', lw=2))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_reservoir_rank(results, save_path):
    """Bar chart of perpendicular reservoir effective rank per condition."""
    conds = list(results.keys())
    ranks = [results[c]['grassmann']['reservoir_effective_rank'] for c in conds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(conds)), ranks)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha='right')
    ax.set_ylabel('Effective rank (exp(entropy))')
    ax.set_title('Perpendicular Reservoir Effective Rank')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_transfer_accuracy(results, save_path):
    """Bar chart of transfer accuracy to the held-out task."""
    conds = list(results.keys())
    accs = [results[c]['transfer']['best_accuracy'] for c in conds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(conds)), accs)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha='right')
    ax.set_ylabel('Task 6 test accuracy (best during transfer training)')
    ax.set_title('Transfer to Held-Out Magnitude Bucket Task')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_per_task_performance(results, save_path):
    """Conditions x Tasks heatmap of final test metric."""
    conds = list(results.keys())
    tasks = TRAINED_TASKS
    mat = np.full((len(conds), len(tasks)), np.nan)
    for i, c in enumerate(conds):
        per_task = results[c].get('final_test', {})
        for j, t in enumerate(tasks):
            if t in per_task:
                vals = list(per_task[t].values())
                mat[i, j] = vals[0] if vals else np.nan
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect='auto', cmap='plasma')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.set_title('Per-task test metric (conditions x tasks)')
    for i in range(len(conds)):
        for j in range(len(tasks)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center',
                        color='white', fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_per_primitive(results, save_path):
    """Two-panel plot: off-diagonal energy after permutation vs after mixer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cond, r in results.items():
        pp = r['grassmann']['per_primitive_occupancy']
        axes[0].plot(range(1, len(pp['post_permute']) + 1), pp['post_permute'],
                     marker='o', label=cond)
        axes[1].plot(range(1, len(pp['post_mix']) + 1), pp['post_mix'],
                     marker='o', label=cond)
    axes[0].set_title('Off-diag energy after permutation')
    axes[1].set_title('Off-diag energy after mixer')
    for ax in axes:
        ax.set_xlabel('Layer')
        ax.set_ylabel('Off-diagonal fraction')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


def plot_seq_phase_trajectory(phase_results, single_task_ranks, save_path):
    """Staircase plot of reservoir rank across G_seq curriculum phases.

    phase_results: list of per-seed phase snapshot lists, each snapshot has
        'reservoir_effective_rank', 'new_task', 'active_tasks'.
    single_task_ranks: dict task_name -> mean rank from single-task runs.
    """
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    task_labels = {
        'spatial': 'D_spa', 'classification': 'A_cls',
        'addition': 'B_add', 'odd_even': 'E_oe', 'comparison': 'C_cmp',
    }
    phase_order = SEQUENTIAL_ORDER

    ax = axes[0]
    all_ranks = []
    for seed_data in phase_results:
        ranks = [snap['reservoir_effective_rank'] for snap in seed_data]
        all_ranks.append(ranks)
        phases = list(range(1, len(ranks) + 1))
        ax.plot(phases, ranks, marker='o', alpha=0.5, linewidth=1.2)

    if all_ranks:
        mean_ranks = np.mean(all_ranks, axis=0)
        ax.plot(range(1, len(mean_ranks) + 1), mean_ranks,
                marker='o', color='black', linewidth=2.5, label='mean')

        prev = 0.0
        for i, (r, task) in enumerate(zip(mean_ranks, phase_order)):
            delta = r - prev
            ax.annotate(f'+{delta:.2f}',
                        xy=(i + 1, r), xytext=(i + 1.05, r - 0.15),
                        fontsize=8, color='black')
            prev = r

    xtick_labels = [f'Phase {i+1}\n+{task_labels.get(t, t)}'
                    for i, t in enumerate(phase_order)]
    ax.set_xticks(range(1, len(phase_order) + 1))
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel('Reservoir effective rank')
    ax.set_title('G_seq: reservoir rank trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    if all_ranks:
        mean_ranks_arr = np.mean(all_ranks, axis=0)
        deltas = [mean_ranks_arr[0]] + [mean_ranks_arr[i] - mean_ranks_arr[i-1]
                                         for i in range(1, len(mean_ranks_arr))]
        colors = plt.cm.tab10(np.linspace(0, 1, len(phase_order)))
        for i, (task, delta, c) in enumerate(zip(phase_order, deltas, colors)):
            st_rank = single_task_ranks.get(task, None)
            label = task_labels.get(task, task)
            ax2.scatter([label], [delta], color=c, s=120, zorder=3,
                        label=f'delta ({label})')
            if st_rank is not None:
                ax2.scatter([label], [st_rank], color=c, s=120, marker='^',
                            zorder=3, label=f'single-task ({label})')
                ax2.plot([label, label], [delta, st_rank], color=c,
                         linewidth=1.0, linestyle='--', alpha=0.6)

    ax2.set_ylabel('Effective rank')
    ax2.set_title('Delta rank per phase vs single-task baseline\n'
                  'circle=delta, triangle=single-task')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  wrote {save_path}")


# ============================================================
# Seed aggregation helper
# ============================================================

def _aggregate_across_seeds(all_results, conditions, seeds):
    """Collapse (cond, seed) -> cond by averaging Grassmannian / final metrics."""
    agg = {}
    for cond in conditions:
        seed_runs = [all_results[f"{cond}_s{s}"] for s in seeds
                     if f"{cond}_s{s}" in all_results]
        if not seed_runs:
            continue

        g0 = seed_runs[0]['grassmann']
        avg_layer_occ = np.mean(
            [np.array(r['grassmann']['layer_occupancy']) for r in seed_runs], axis=0
        )
        avg_meaning = {}
        for t in g0['meaning_per_task']:
            vals = [np.array(r['grassmann']['meaning_per_task'].get(t, []))
                    for r in seed_runs if t in r['grassmann']['meaning_per_task']]
            if vals and len(vals[0]) > 0:
                avg_meaning[t] = np.mean(vals, axis=0).tolist()
        avg_rank = float(np.mean(
            [r['grassmann']['reservoir_effective_rank'] for r in seed_runs]
        ))
        avg_pp = {
            'post_permute': np.mean(
                [np.array(r['grassmann']['per_primitive_occupancy']['post_permute'])
                 for r in seed_runs], axis=0
            ).tolist(),
            'post_mix': np.mean(
                [np.array(r['grassmann']['per_primitive_occupancy']['post_mix'])
                 for r in seed_runs], axis=0
            ).tolist(),
        }
        avg_transfer = float(np.mean(
            [r['transfer']['best_accuracy'] for r in seed_runs]
        ))

        final_test = {}
        for t in TRAINED_TASKS:
            metric_keys_seen = set()
            for r in seed_runs:
                if t in r.get('final_test', {}):
                    metric_keys_seen.update(r['final_test'][t].keys())
            if metric_keys_seen:
                final_test[t] = {}
                for mk in metric_keys_seen:
                    vals = [r['final_test'][t][mk] for r in seed_runs
                            if t in r.get('final_test', {}) and mk in r['final_test'][t]]
                    if vals:
                        final_test[t][mk] = float(np.mean(vals))

        agg[cond] = {
            'grassmann': {
                'layer_occupancy': avg_layer_occ.tolist(),
                'meaning_per_task': avg_meaning,
                'reservoir_effective_rank': avg_rank,
                'per_primitive_occupancy': avg_pp,
            },
            'transfer': {'best_accuracy': avg_transfer},
            'final_test': final_test,
        }
    return agg


# ============================================================
# Batch plot driver
# ============================================================

def _make_plots(aggregated, save_dir='.', prefix='exp7_',
                all_results=None, seeds=None):
    """Generate all standard figures into save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, prefix)

    plot_latent_occupancy(aggregated, base + 'latent_occupancy.png')
    plot_meaning_heatmap(aggregated, base + 'meaning_heatmap.png')
    plot_meaning_per_layer_heatmap(aggregated, base + 'meaning_per_layer.png')
    plot_reservoir_rank(aggregated, base + 'reservoir_rank.png')
    plot_transfer_accuracy(aggregated, base + 'transfer_accuracy.png')
    plot_per_task_performance(aggregated, base + 'per_task_performance.png')
    plot_per_primitive(aggregated, base + 'per_primitive.png')

    if all_results is not None and seeds is not None:
        phase_results = []
        for s in seeds:
            key = f'G_seq_s{s}'
            if key in all_results:
                snaps = all_results[key].get('history', {}).get('phase_snapshots', [])
                if snaps:
                    phase_results.append(snaps)
        if phase_results:
            task_to_cond = {
                'spatial': 'D_spa', 'classification': 'A_cls',
                'addition': 'B_add', 'odd_even': 'E_oe', 'comparison': 'C_cmp',
            }
            single_task_ranks = {}
            for task, cond in task_to_cond.items():
                if cond in aggregated:
                    single_task_ranks[task] = (
                        aggregated[cond]['grassmann']['reservoir_effective_rank']
                    )
            plot_seq_phase_trajectory(
                phase_results, single_task_ranks, base + 'seq_phase_trajectory.png'
            )

    print(f"Figures written to {save_dir}/")


# ============================================================
# CLI entry point: python plots.py <results_json>
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regenerate all figures from a saved results JSON file.'
    )
    parser.add_argument('results_json', help='Path to exp7_results_*.json')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for figures (default: same as JSON)')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help='Seeds used in the experiment (default: auto-detect)')
    args = parser.parse_args()

    results_path = args.results_json
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # Auto-detect conditions and seeds from the keys
    conditions_seen = set()
    seeds_seen = set()
    for key in all_results:
        # Keys are like "A_cls_s0", "F_multi_s2", etc.
        parts = key.rsplit('_s', 1)
        if len(parts) == 2:
            try:
                s = int(parts[1])
                seeds_seen.add(s)
                conditions_seen.add(parts[0])
            except ValueError:
                pass

    seeds = args.seeds if args.seeds is not None else sorted(seeds_seen)
    conditions = sorted(conditions_seen)

    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")

    aggregated = _aggregate_across_seeds(all_results, conditions, seeds)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(results_path))
    _make_plots(aggregated, save_dir=out_dir,
                all_results=all_results, seeds=seeds)
