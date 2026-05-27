# Multi-Task MNIST Geodesic Coverage -- Code Supplement

## Overview

This code accompanies [paper title]. We train a neural backbone on 7 multi-task configurations of MNIST and measure the Grassmannian reservoir rank -- the effective dimensionality of the backbone's output representation -- as an empirical proxy for how many independent task-relevant subspaces the network develops. The main finding is that reservoir rank tracks task geometry: tasks solvable via scalar shortcuts converge to rank-1, spatial tasks reach rank~2, and classification-based tasks fill rank 4-6. Multi-task training fills more geodesic directions than any single task.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

```bash
# Full experiment (all 7 conditions, 3 seeds each, ~50 epochs):
python run_experiment.py

# Single condition:
python run_experiment.py --conditions B_add --seeds 0 1 2

# 200-epoch spatial probe run (B_add only):
python run_experiment.py --conditions B_add --seeds 0 \
    --epochs 200 --probe-epochs 50 100 150 200 \
    --profile-weight 0.5 --rank-weight 0.1 --rank-target 4.0 \
    --perturb-lr-mult 10.0 --run-id long_probe

# G_seq with per-phase reservoir snapshots:
python run_experiment.py --conditions G_seq --seeds 0 1 2 \
    --seq-phase-snapshots --seq-epochs-per-phase 50 --run-id gseq_phases

# Regenerate figures from saved results:
python plots.py results/exp7_results_20260515_212500.json
```

## Conditions

| Condition | Training | Tasks |
|-----------|----------|-------|
| A_cls | Single-task | classification only |
| B_add | Single-task | addition (two-digit sum) |
| C_cmp | Single-task | comparison (left > right?) |
| D_spa | Single-task | spatial center-of-mass |
| E_oe | Single-task | odd/even parity |
| F_multi | Joint multi-task | all 5 tasks simultaneously |
| G_seq | Sequential curriculum | spatial -> cls -> add -> oe -> cmp |

## Key results

- Rank-1: B_add, C_cmp, E_oe (scalar shortcuts)
- Rank 1-2: D_spa (2D spatial output)
- Rank 4-6: A_cls, F_multi, G_seq (10-class or multi-task diversity)
- D_spa paradox: higher rank than B_add but lower transfer accuracy (44% vs 70%) -- semantic alignment with the transfer task matters as much as rank
- G_seq debris field: sequential curriculum fills rank comparably to multi-task, but meaning fractions skew toward the most recently trained task

## Architecture (H_rotational)

The backbone uses three primitive types per layer: (1) LearnedButterflyPermutation for routing, (2) RotationalProjectionMixer (A = theta*R_base + eps*A_perturb, M = exp(A)) for pairwise mixing, (3) SparseDensePreconditioner for input normalization. Geometric measurements use SVD-entropy effective rank and per-layer principal angles.

## Note on GramBlock architecture

> **Note on GramBlock architecture**: The `GramBlockDiagonalMixer` computes a `state_dim x state_dim` feature Gram matrix as a global pre-mixing step before the rotational layers. At `state_dim=3` (intensity, norm_x, norm_y) used in these experiments, the Gram matrix is 3x3 and adds minimal signal over the direct path -- the model learned this, with the Gram-mixed path receiving only ~28-32% of the weight of the direct path (A2/A1 ratio). The GramBlock architecture is included for completeness and may be more meaningful at higher state dimensions where the Gram matrix captures richer feature correlations.

## Citation

[to be filled]
