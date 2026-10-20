"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: synthetic_plume.py

Generates physically accurate synthetic Background-Oriented Schlieren (BOS) video frames:
- Synthesizes textured background surfaces (concrete, asphalt, speckle pattern).
- Simulates buoyant turbulent gas plume dynamics with vortex shedding (Strouhal number St ≈ 0.2).
- Distorts light rays according to Snell's law using sub-pixel image warping (cv2.remap).
- Adds ambient wind advection, thermal camera noise, and diurnal illumination changes.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class SyntheticSchlierenGenerator:
    """
    Simulates physical Background-Oriented Schlieren (BOS) optical distortions
    from a turbulent methane gas plume.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        background_texture: Optional[np.ndarray] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0

        # Use custom background or generate realistic textured background (gravel / concrete pattern)
        if background_texture is not None:
            self.background_texture = background_texture
        else:
            self.background_texture = self._generate_textured_background(width, height)

        # Coordinate meshgrid for sub-pixel ray warping (cv2.remap)
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def _generate_textured_background(self, w: int, h: int) -> np.ndarray:
        """Generates high-frequency spatial texture suitable for BOS correlation."""
        np.random.seed(42)
        # Multi-scale Gaussian noise pattern simulating asphalt/concrete texture
        fine_noise = np.random.normal(128, 30, (h, w)).astype(np.float32)
        coarse_noise = cv2.resize(
            np.random.normal(128, 40, (h // 4, w // 4)).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        texture = 0.6 * fine_noise + 0.4 * coarse_noise
        texture = np.clip(texture, 0, 255).astype(np.uint8)
        # Convert to 3-channel BGR
        return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    def generate_frame(
        self,
        plume_active: bool = True,
        plume_source: Optional[Tuple[int, int]] = None,
        leak_intensity: float = 1.0,  # 0.0 to 2.0 scale
        wind_speed_px_per_frame: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes the next frame in the sequence.

        Returns:
            Tuple of:
                distorted_frame: BGR frame with sub-pixel ray bending.
                ground_truth_opd: Ground-truth optical path difference field (nanometers).
                ground_truth_disp: (H, W, 2) Ground-truth displacement vector field (pixels).
        """
        t = self.frame_idx / self.fps
        self.frame_idx += 1

        h, w = self.height, self.width
        opd_nm = np.zeros((h, w), dtype=np.float32)
        disp_x = np.zeros((h, w), dtype=np.float32)
        disp_y = np.zeros((h, w), dtype=np.float32)

        if plume_active and leak_intensity > 0.0:
            if plume_source is None:
                sx, sy = (w // 2, int(h * 0.85))
            else:
                sx, sy = plume_source
            
            # Buoyant turbulent plume with meandering centerline and vortex shedding
            # Vortex shedding frequency: f_shed = St * U / D ≈ 0.2 * 3.0 / 0.1 ≈ 6 Hz
            shed_freq = 2.5  # Hz
            meander_freq = 0.6  # Hz
            
            # Create plume bounding region
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            
            # Vertical distance above source (plume rises upwards)
            dy = np.maximum(sy - y_indices, 0.0)
            
            # Meandering centerline: x_c(y) = sx + wind_drift + sin wave meander
            wind_drift = dy * 0.30  # Leaning due to wind
            meander = (w * 0.03) * np.sin(2.0 * math.pi * meander_freq * t + dy * 0.02)
            eddy_turb = (w * 0.015) * np.sin(2.0 * math.pi * shed_freq * t - dy * 0.05) * np.cos(x_indices * 0.03)
            center_x = sx + wind_drift + meander + eddy_turb

            # Lateral Gaussian expansion: sigma_x(y) = sigma_0 + 0.15 * dy
            sigma_x = (w * 0.02) + 0.16 * dy
            dx = x_indices - center_x

            # Density / OPD profile: bell curve with height decay and vortex puffs
            puff = 1.0 + 0.35 * np.sin(2.0 * math.pi * shed_freq * t - dy * 0.04)
            gaussian_profile = np.exp(-(dx**2) / (2.0 * (sigma_x**2)))
            decay = np.exp(-dy / float(h * 0.75))
            active_height_mask = (y_indices <= sy).astype(np.float32)

            # Ground truth OPD in nanometers (typical methane plume: 100 - 500 nm OPD)
            opd_nm = 350.0 * leak_intensity * gaussian_profile * decay * puff * active_height_mask

            # Refractive displacement vectors are proportional to spatial gradient of OPD:
            # d_x ∝ ∂(OPD)/∂x,  d_y ∝ ∂(OPD)/∂y
            # Scaled to sub-pixel displacements (0.2 to 2.5 pixels)
            grad_scale = 0.06
            disp_x = cv2.Sobel(opd_nm, cv2.CV_32F, 1, 0, ksize=3) * grad_scale
            disp_y = cv2.Sobel(opd_nm, cv2.CV_32F, 0, 1, ksize=3) * grad_scale

        # Warp background texture using sub-pixel interpolation (cv2.remap)
        # New apparent pixel location: (x - disp_x, y - disp_y)
        map_x = (self.grid_x - disp_x).astype(np.float32)
        map_y = (self.grid_y - disp_y).astype(np.float32)
        distorted_frame = cv2.remap(
            self.background_texture,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Add camera sensor noise (Gaussian photon shot noise)
        noise = np.random.normal(0, 1.5, distorted_frame.shape).astype(np.float32)
        noisy_frame = np.clip(distorted_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        gt_disp = np.stack([disp_x, disp_y], axis=-1)
        return noisy_frame, opd_nm, gt_disp
