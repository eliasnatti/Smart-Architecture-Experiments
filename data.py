"""
Data preparation for multi-task MNIST geodesic experiment.

Constants, task specifications, dataset wrappers, and data loaders for
the seven training conditions (A_cls through G_seq). All tasks share a
single 20x40 canvas layout:
  - Single-image tasks: MNIST digit in the LEFT half, right half zeros.
  - Pairwise tasks: digit A in the left half, digit B in the right half.

Tasks:
  classification  -- 10-way CE
  addition        -- MSE on normalized sum (0..18 -> 0..1)
  comparison      -- BCE on (left > right)
  spatial         -- MSE on normalized center-of-mass (cx, cy)
  odd_even        -- BCE on parity
  magnitude_bucket -- CE on small/medium/large (HELD OUT for transfer)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ============================================================
# Canvas geometry constants
# ============================================================

MNIST_CROP_OFFSET = 4                          # center-crop 28x28 -> 20x20
MNIST_CROP_SIZE = 20
CANVAS_H = MNIST_CROP_SIZE                     # 20
CANVAS_W = MNIST_CROP_SIZE * 2                 # 40 (left|right halves)
NUM_STATE_VECTORS = CANVAS_H * CANVAS_W        # 800

# Training defaults
BATCH_SIZE = 64

# ============================================================
# Task specifications
# ============================================================

def _magnitude_bucket(label):
    """Small (0-3) -> 0, medium (4-6) -> 1, large (7-9) -> 2."""
    if label <= 3:
        return 0
    elif label <= 6:
        return 1
    else:
        return 2


TASK_SPECS = {
    'classification': {
        'kind': 'single',
        'output_dim': 10,
        'loss': 'ce',
        'metric': 'accuracy',
    },
    'addition': {
        'kind': 'pairwise',
        'output_dim': 1,
        'loss': 'mse',
        'metric': 'mae',
        # Target range is [0, 18] (sum of two MNIST labels). Normalized to
        # [0, 1] during training; un-scaled when reporting MAE.
        'target_scale': 18.0,
    },
    'comparison': {
        'kind': 'pairwise',
        'output_dim': 1,
        'loss': 'bce',
        'metric': 'accuracy',
    },
    'spatial': {
        'kind': 'single',
        'output_dim': 2,
        'loss': 'mse',
        'metric': 'mse',
    },
    'odd_even': {
        'kind': 'single',
        'output_dim': 1,
        'loss': 'bce',
        'metric': 'accuracy',
    },
    'magnitude_bucket': {
        'kind': 'single',
        'output_dim': 3,
        'loss': 'ce',
        'metric': 'accuracy',
    },
}

TRAINED_TASKS = ['classification', 'addition', 'comparison', 'spatial', 'odd_even']
HELD_OUT_TASK = 'magnitude_bucket'
ALL_TASKS = TRAINED_TASKS + [HELD_OUT_TASK]

# Sequential curriculum order (Task 4 -> 1 -> 2 -> 5 -> 3)
SEQUENTIAL_ORDER = ['spatial', 'classification', 'addition', 'odd_even', 'comparison']

# Task weights for Condition F (multi-task): all 1.0 (targets are normalized)
TASK_WEIGHTS = {
    'classification': 1.0,
    'addition': 1.0,
    'comparison': 1.0,
    'spatial': 1.0,
    'odd_even': 1.0,
}

# ============================================================
# Image preprocessing
# ============================================================

def crop_mnist(batch_images):
    """Center-crop MNIST 28x28 -> 20x20."""
    o = MNIST_CROP_OFFSET
    s = MNIST_CROP_SIZE
    if batch_images.dim() == 4:
        return batch_images[:, :, o:o+s, o:o+s].contiguous()
    return batch_images[:, o:o+s, o:o+s].contiguous()


def make_canvas_single(images):
    """Place cropped 20x20 images into the LEFT half of a 20x40 canvas.

    images: (B, 1, 20, 20) or (B, 20, 20)
    returns: (B, 1, 20, 40) with right half zeros.
    """
    if images.dim() == 3:
        images = images.unsqueeze(1)
    B = images.size(0)
    canvas = torch.zeros(B, 1, CANVAS_H, CANVAS_W,
                         device=images.device, dtype=images.dtype)
    canvas[:, :, :, :MNIST_CROP_SIZE] = images
    return canvas


def make_canvas_pair(images_a, images_b):
    """Place two cropped 20x20 images side-by-side on a 20x40 canvas."""
    if images_a.dim() == 3:
        images_a = images_a.unsqueeze(1)
    if images_b.dim() == 3:
        images_b = images_b.unsqueeze(1)
    B = images_a.size(0)
    canvas = torch.zeros(B, 1, CANVAS_H, CANVAS_W,
                         device=images_a.device, dtype=images_a.dtype)
    canvas[:, :, :, :MNIST_CROP_SIZE] = images_a
    canvas[:, :, :, MNIST_CROP_SIZE:] = images_b
    return canvas


def compute_center_of_mass_batch(images):
    """Compute normalized (cx, cy) center of mass per 20x20 image.

    images: (B, 1, 20, 20) or (B, 20, 20) with values in [0, 1]
    returns: (B, 2) in [0, 1]
    """
    if images.dim() == 4:
        images = images.squeeze(1)
    B, H, W = images.shape
    total = images.sum(dim=(1, 2)) + 1e-8
    y_grid = torch.arange(H, dtype=torch.float32, device=images.device).view(1, H, 1)
    x_grid = torch.arange(W, dtype=torch.float32, device=images.device).view(1, 1, W)
    cy = (images * y_grid).sum(dim=(1, 2)) / total / (H - 1)
    cx = (images * x_grid).sum(dim=(1, 2)) / total / (W - 1)
    return torch.stack([cx, cy], dim=1)


# ============================================================
# Dataset wrapper
# ============================================================

class PairwiseMNIST(Dataset):
    """Wraps MNIST to yield pairs for addition/comparison tasks."""

    def __init__(self, mnist_dataset, length=None, seed=None):
        self.base = mnist_dataset
        self.length = length if length is not None else len(mnist_dataset)
        rng = np.random.default_rng(seed)
        self.idx_a = rng.integers(0, len(mnist_dataset), size=self.length)
        self.idx_b = rng.integers(0, len(mnist_dataset), size=self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        img_a, lbl_a = self.base[int(self.idx_a[i])]
        img_b, lbl_b = self.base[int(self.idx_b[i])]
        return img_a, lbl_a, img_b, lbl_b


# ============================================================
# Dataset and loader construction
# ============================================================

def get_mnist_datasets():
    """Download MNIST to <repo>/data/ and return (train_ds, test_ds)."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def build_loaders(train_ds, test_ds, batch_size=BATCH_SIZE, seed=0):
    """Build one loader per task kind.

    Single-image tasks share the standard MNIST train/test loaders.
    Pairwise tasks use a PairwiseMNIST wrapper with a fixed pair index.

    Returns dict with keys:
      'single_train', 'single_test', 'pair_train', 'pair_test'
    """
    single_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    single_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    pair_train_ds = PairwiseMNIST(train_ds, length=len(train_ds), seed=seed)
    pair_test_ds = PairwiseMNIST(test_ds, length=len(test_ds), seed=seed + 10000)
    pair_train = DataLoader(pair_train_ds, batch_size=batch_size, shuffle=True)
    pair_test = DataLoader(pair_test_ds, batch_size=batch_size, shuffle=False)

    return {
        'single_train': single_train,
        'single_test': single_test,
        'pair_train': pair_train,
        'pair_test': pair_test,
    }


def prepare_task_batch(task_name, raw_batch, device='cpu'):
    """Convert a raw loader batch into (canvas, target) for the given task.

    raw_batch comes from:
      - single loader: (images, labels)
      - pair loader:   (images_a, labels_a, images_b, labels_b)

    Returns: (canvas (B,1,20,40), target_tensor)
    """
    spec = TASK_SPECS[task_name]

    if spec['kind'] == 'single':
        images, labels = raw_batch
        images = images.to(device)
        labels = labels.to(device)
        cropped = crop_mnist(images)
        canvas = make_canvas_single(cropped)

        if task_name == 'classification':
            target = labels.long()
        elif task_name == 'spatial':
            target = compute_center_of_mass_batch(cropped)
        elif task_name == 'odd_even':
            target = (labels % 2).float().unsqueeze(1)
        elif task_name == 'magnitude_bucket':
            bucketed = torch.tensor(
                [_magnitude_bucket(int(l.item())) for l in labels],
                dtype=torch.long, device=device,
            )
            target = bucketed
        else:
            raise ValueError(f"Unknown single-image task: {task_name}")
        return canvas, target

    elif spec['kind'] == 'pairwise':
        img_a, lbl_a, img_b, lbl_b = raw_batch
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        lbl_a = lbl_a.to(device).float()
        lbl_b = lbl_b.to(device).float()
        cropped_a = crop_mnist(img_a)
        cropped_b = crop_mnist(img_b)
        canvas = make_canvas_pair(cropped_a, cropped_b)

        if task_name == 'addition':
            target = (lbl_a + lbl_b).unsqueeze(1)          # (B, 1) in [0, 18]
            scale = spec.get('target_scale')
            if scale is not None:
                target = target / scale                    # normalize to [0, 1]
        elif task_name == 'comparison':
            target = (lbl_a > lbl_b).float().unsqueeze(1)
        else:
            raise ValueError(f"Unknown pairwise task: {task_name}")
        return canvas, target

    else:
        raise ValueError(f"Unknown task kind: {spec['kind']}")


def iter_task_batches(task_name, loaders):
    """Yield raw batches from the appropriate loader for the task."""
    spec = TASK_SPECS[task_name]
    if spec['kind'] == 'single':
        return iter(loaders['single_train'])
    else:
        return iter(loaders['pair_train'])
