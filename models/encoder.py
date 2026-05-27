"""
Unified State Vector Encoder.

Converts 2D images to unified state vectors s_i in R^d where each state
vector contains both spatial position and intensity as unified features:
    s_i = [intensity_i, norm_x_i, norm_y_i]

Position is explicit (not implicit in indices), all dimensions normalized
to [0, 1], and the representation is dense -- every pixel gets a state vector.
"""

import torch
import torch.nn as nn


class UnifiedStateVectorEncoder:
    """Encodes 2D images as unified state vectors.

    Each pixel at (x, y) with intensity v produces a state vector:
        s_i = [v, x/(W-1), y/(H-1)]

    Args:
        encoding_strategy: type of encoding (currently only "dense_position_intensity")
        image_height: height of input images
        image_width: width of input images
        state_dim: dimensionality of each state vector (must be >= 3)
    """

    def __init__(self, encoding_strategy: str = "dense_position_intensity",
                 image_height: int = 28, image_width: int = 28,
                 state_dim: int = 3):
        if state_dim < 3:
            raise ValueError("State dimension must be at least 3 for [intensity, x, y]")

        self.encoding_strategy = encoding_strategy
        self.image_height = image_height
        self.image_width = image_width
        self.state_dim = state_dim
        self.num_state_vectors = image_height * image_width

        self._precompute_position_coordinates()

    def _precompute_position_coordinates(self):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.image_height, dtype=torch.float32),
            torch.arange(self.image_width, dtype=torch.float32),
            indexing='ij',
        )
        self.norm_x = x_coords / max(self.image_width - 1, 1)
        self.norm_y = y_coords / max(self.image_height - 1, 1)
        self.norm_x_flat = self.norm_x.flatten()
        self.norm_y_flat = self.norm_y.flatten()

    def encode_mnist(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to unified state vectors.

        Args:
            images: (B, 1, H, W) or (B, H, W)
        Returns:
            state_vectors: (B, N, d) where N = H*W, d = state_dim
        """
        if images.dim() == 3:
            images = images.unsqueeze(1)
        elif images.dim() == 4 and images.size(1) == 1:
            pass
        else:
            raise ValueError(
                f"Expected images with shape (B,H,W) or (B,1,H,W), got {images.shape}"
            )

        batch_size = images.size(0)
        intensities_flat = images.squeeze(1).view(batch_size, -1)

        norm_x_flat = self.norm_x_flat.to(images.device)
        norm_y_flat = self.norm_y_flat.to(images.device)

        state_vectors = torch.zeros(
            batch_size, self.num_state_vectors, self.state_dim,
            device=images.device, dtype=images.dtype,
        )
        state_vectors[:, :, 0] = intensities_flat
        state_vectors[:, :, 1] = norm_x_flat.unsqueeze(0).expand(batch_size, -1)
        state_vectors[:, :, 2] = norm_y_flat.unsqueeze(0).expand(batch_size, -1)
        return state_vectors

    def decode_to_images(self, state_vectors: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from state vectors (intensity dimension only)."""
        batch_size = state_vectors.size(0)
        intensities = state_vectors[:, :, 0]
        return intensities.view(batch_size, 1, self.image_height, self.image_width)
