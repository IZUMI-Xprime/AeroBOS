"""
AeroBOS: Core Background-Oriented Schlieren (BOS) Physics Engine
Module: aerobos/bos_core.py

Implements BOSCore:
- Dense optical flow computation via OpenCV DISOpticalFlow.
- Spatial divergence calculation of displacement vector fields (du/dx + dv/dy).
- Regularized 2D Poisson solver in Fourier space (np.fft.fft2 / np.fft.ifft2)
  to reconstruct the scalar Optical Path Difference (OPD) field.
"""

from __future__ import annotations

import cv2
import numpy as np


class BOSCore:
    """
    Core Background-Oriented Schlieren (BOS) optical flow and Poisson inversion engine.
    """

    def __init__(
        self,
        preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        regularization: float = 1e-4,
    ) -> None:
        """
        Initializes the DIS Optical Flow solver and Fourier Poisson parameters.

        Args:
            preset: DISOpticalFlow preset (PRESET_ULTRAFAST, PRESET_FAST, or PRESET_MEDIUM).
            regularization: Tikhonov regularization parameter for Laplacian pseudo-inverse.
        """
        self.dis = cv2.DISOpticalFlow_create(preset)
        self.regularization = regularization

    def calc_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """
        Calculates dense sub-pixel optical flow between two grayscale frames.

        Args:
            prev_gray: Reference or previous grayscale image (uint8, HxW).
            curr_gray: Current grayscale image (uint8, HxW).

        Returns:
            flow: (H, W, 2) array of displacement vectors (u, v) in pixels.
        """
        return self.dis.calc(prev_gray, curr_gray, None)

    def compute_divergence(self, flow: np.ndarray) -> np.ndarray:
        """
        Computes the spatial divergence of the flow displacement field:
            div(d) = ∂u/∂x + ∂v/∂y

        Args:
            flow: (H, W, 2) flow array with u=flow[..., 0] and v=flow[..., 1].

        Returns:
            divergence: (H, W) spatial divergence field.
        """
        u = flow[..., 0]
        v = flow[..., 1]
        du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
        dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
        return du_dx + dv_dy

    def solve_poisson_fft(self, divergence: np.ndarray) -> np.ndarray:
        """
        Solves ∇² OPD = divergence in the frequency domain using 2D FFT:
            -(k_x² + k_y²) · F{OPD} = F{divergence}

        Args:
            divergence: (H, W) spatial divergence field.

        Returns:
            opd: (H, W) reconstructed scalar Optical Path Difference field.
        """
        h, w = divergence.shape

        # Compute frequency coordinates
        kx = 2.0 * np.pi * np.fft.fftfreq(w)
        ky = 2.0 * np.pi * np.fft.fftfreq(h)
        kx_grid, ky_grid = np.meshgrid(kx, ky)

        # 5-point discrete Laplacian eigenvalues in Fourier domain
        laplacian_kernel = 2.0 * (np.cos(kx_grid) - 1.0) + 2.0 * (np.cos(ky_grid) - 1.0)

        # Regularized pseudo-inverse transfer function
        denom = laplacian_kernel**2 + self.regularization
        h_inv = laplacian_kernel / denom
        h_inv[0, 0] = 0.0  # Zero DC component (removes arbitrary gauge offset)

        # 2D Fast Fourier Transform
        div_fft = np.fft.fft2(divergence.astype(np.float32))
        opd_fft = div_fft * h_inv

        # Inverse 2D FFT to recover spatial domain scalar OPD
        opd = np.real(np.fft.ifft2(opd_fft))
        return opd.astype(np.float32)
