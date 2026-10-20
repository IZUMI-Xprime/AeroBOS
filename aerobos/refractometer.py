"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: refractometer.py

Implements SchlierenRefractometer:
- Ingests high-resolution video frames (1080p/4K) observing naturally textured backgrounds.
- Continuous background reference modeling with adaptive illumination compensation.
- Sub-pixel dense refractive displacement vector fields using OpenCV 5 DIS Optical Flow.
- 2D Poisson solver via Discrete Fourier Transform (cv2.dft / cv2.idft) on the displacement
  divergence field (∇ · d) to invert Snell's law and reconstruct line-of-sight gas density fields.
- Temporal bandpass filtering isolating turbulent gas convective eddies from environmental wind.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class OpticalGeometry:
    """
    Physical geometric parameters governing Background-Oriented Schlieren (BOS).

    Coordinate Frame:
        Camera Lens at z = 0
        Gas Disturbance Plane at z = Z_D
        Background Texture Plane at z = Z_B
    """
    focal_length_m: float = 0.050          # Camera lens focal length (50mm lens)
    dist_camera_to_gas_m: float = 5.0      # Z_D: Disturbance distance (meters)
    dist_camera_to_bg_m: float = 10.0      # Z_B: Background distance (meters)
    pixel_pitch_m: float = 3.45e-6         # Pixel sensor pitch (3.45 um for Sony Pregius sensor)
    ambient_refractive_index: float = 1.000277  # Air at 20 deg C, 101.3 kPa, 550 nm
    gas_gladstone_dale_k: float = 4.3e-4   # Methane (CH4) Gladstone-Dale constant (m^3 / kg)
    air_gladstone_dale_k: float = 2.26e-4  # Ambient air Gladstone-Dale constant (m^3 / kg)

    @property
    def magnification(self) -> float:
        """Optical magnification M = f / (Z_B - f) ≈ f / Z_B."""
        return self.focal_length_m / (self.dist_camera_to_bg_m - self.focal_length_m)

    @property
    def bos_coupling_coefficient(self) -> float:
        """
        Coupling coefficient C_BOS relating apparent sensor pixel shift d = (u, v)
        to the gradient of the optical path difference (OPD) field Φ(x, y).

        Mathematical Derivation:
        -------------------------
        Paraxial ray deviation angle ε = (1 / n0) ∫ ∇_xy n(x, y, z) dz
        Virtual displacement at background plane: Δx_B = (Z_B - Z_D) * ε
        Projected displacement on camera sensor: d_sensor = M * Δx_B = M * (Z_B - Z_D) * ε
        Converting sensor meters to pixels (divide by pixel pitch p_pix):
            d_pix = [ M * (Z_B - Z_D) / (n0 * p_pix) ] * ∇_xy ∫ (n - n0) dz
            d_pix = C_BOS * ∇_xy Φ(x, y)
        """
        delta_z = self.dist_camera_to_bg_m - self.dist_camera_to_gas_m
        return (self.magnification * delta_z) / (self.ambient_refractive_index * self.pixel_pitch_m)


@dataclass
class RefractometerConfig:
    """Configuration hyperparameters for SchlierenRefractometer."""
    geometry: OpticalGeometry = field(default_factory=OpticalGeometry)
    
    # DIS Optical Flow Parameters (OpenCV 5)
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST  # PRESET_ULTRAFAST, PRESET_FAST, PRESET_MEDIUM
    dis_finest_scale: int = 1
    dis_patch_size: int = 8
    dis_patch_stride: int = 4
    dis_gradient_descent_iters: int = 16
    dis_variational_alpha: float = 0.05
    dis_variational_gamma: float = 5.0
    
    # Background Model & Illumination Compensation
    bg_learning_rate: float = 0.015         # Exponential moving average learning rate
    illumination_norm_kernel: int = 41      # Local contrast normalization kernel size (odd)
    freeze_bg_on_anomaly: bool = True       # Prevent turbulent gas from corrupting background
    
    # 2D Poisson Solver
    poisson_regularization: float = 1e-4    # Tikhonov regularization λ for Poisson denominator
    use_optimal_dft_padding: bool = True    # Pad image to 2^p * 3^q * 5^r for fastest FFT
    
    # Temporal Bandpass Filter (isolates gas turbulence from DC wind / thermal drift)
    fps: float = 30.0                       # Video frame rate
    bandpass_low_hz: float = 0.5            # High-pass cutoff (filters slow wind / shadows)
    bandpass_high_hz: float = 8.0           # Low-pass cutoff (filters high-frequency camera noise)
    
    # Processing downscale factor (1.0 = native 1080p/4K; 0.5 = 2x downscale for real-time edge)
    processing_scale: float = 1.0


class SchlierenRefractometer:
    """
    High-Performance Background-Oriented Schlieren (BOS) Refractometer.

    Extracts sub-pixel refractive displacement fields caused by gas density gradients
    and reconstructs 2D integrated density / concentration maps via Fourier Poisson inversion.
    """

    def __init__(self, config: Optional[RefractometerConfig] = None) -> None:
        self.config = config or RefractometerConfig()
        
        # Initialize OpenCV 5 DIS Optical Flow engine
        self._dis_flow = cv2.DISOpticalFlow.create(self.config.dis_preset)
        self._dis_flow.setFinestScale(self.config.dis_finest_scale)
        self._dis_flow.setPatchSize(self.config.dis_patch_size)
        self._dis_flow.setPatchStride(self.config.dis_patch_stride)
        self._dis_flow.setGradientDescentIterations(self.config.dis_gradient_descent_iters)
        self._dis_flow.setVariationalRefinementAlpha(self.config.dis_variational_alpha)
        self._dis_flow.setVariationalRefinementGamma(self.config.dis_variational_gamma)
        
        # Adaptive background state
        self._bg_reference: Optional[np.ndarray] = None
        self._bg_variance: Optional[np.ndarray] = None
        self._frame_count: int = 0
        
        # Frequency domain precomputed Laplacian kernels
        self._dft_h: int = 0
        self._dft_w: int = 0
        self._laplacian_kernel_dft: Optional[np.ndarray] = None
        
        # Temporal bandpass filter state (recursive 2nd-order IIR biquad)
        self._biquad_init: bool = False
        self._b0: float = 0.0
        self._b1: float = 0.0
        self._b2: float = 0.0
        self._a1: float = 0.0
        self._a2: float = 0.0
        self._w1: Optional[np.ndarray] = None
        self._w2: Optional[np.ndarray] = None
        self._setup_temporal_bandpass()

        # Profiling telemetry
        self.last_profile_ms: Dict[str, float] = {
            "preprocess": 0.0,
            "optical_flow": 0.0,
            "divergence": 0.0,
            "poisson_solve": 0.0,
            "temporal_filter": 0.0,
            "total": 0.0,
        }

    # =========================================================================
    # Mathematical Physics: Illumination Normalization & Background Model
    # =========================================================================
    def _normalize_illumination(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Applies local contrast normalization to isolate surface texture from
        spatially varying illumination (shadows, diurnal sunlight shifts).

        I_norm(x, y) = [ I(x, y) - μ_local(x, y) ] / [ σ_local(x, y) + ε ]

        Optimized with separable box/Gaussian filters for Arm Neon / SVE SIMD execution.
        """
        ksize = self.config.illumination_norm_kernel
        if ksize % 2 == 0:
            ksize += 1

        img_f = gray_frame.astype(np.float32)
        local_mean = cv2.boxFilter(img_f, cv2.CV_32F, (ksize, ksize), normalize=True)
        img_sq = img_f * img_f
        local_sq_mean = cv2.boxFilter(img_sq, cv2.CV_32F, (ksize, ksize), normalize=True)
        local_var = np.maximum(local_sq_mean - (local_mean * local_mean), 1.0)
        local_std = np.sqrt(local_var)

        norm = (img_f - local_mean) / (local_std + 1e-3)
        # Scaled back to 8-bit unsigned dynamic range [0, 255] for DIS flow vectorization
        norm_uint8 = np.clip(norm * 40.0 + 128.0, 0, 255).astype(np.uint8)
        return norm_uint8

    def _update_background(self, norm_frame: np.ndarray, anomaly_mask: Optional[np.ndarray]) -> None:
        """
        Continuously updates background reference model using an exponential moving average.
        When gas anomaly is detected, updates are frozen in the plume region to prevent
        ghosting or background absorption of the refractive disturbance.
        """
        norm_f = norm_frame.astype(np.float32)
        if self._bg_reference is None:
            self._bg_reference = norm_f.copy()
            self._bg_variance = np.zeros_like(norm_f)
            return

        alpha = self.config.bg_learning_rate
        if anomaly_mask is not None and self.config.freeze_bg_on_anomaly:
            # Mask = 1 where anomaly detected (freeze update -> effective alpha = 0)
            update_weight = alpha * (1.0 - anomaly_mask.astype(np.float32))
        else:
            update_weight = alpha

        diff = norm_f - self._bg_reference
        self._bg_reference += update_weight * diff
        if self._bg_variance is not None:
            self._bg_variance = (1.0 - update_weight) * self._bg_variance + update_weight * (diff * diff)

    # =========================================================================
    # Mathematical Physics: 2D Poisson Solver via Discrete Fourier Transform
    # =========================================================================
    def _precompute_poisson_kernel(self, h: int, w: int) -> None:
        """
        Precomputes the 2D discrete Laplacian Fourier spectrum for Poisson inversion:
            ∇² Φ(x, y) = f(x, y)
            -(k_x² + k_y²) · F{Φ}(k_x, k_y) = F{f}(k_x, k_y)

        Discrete grid representation of Laplacian with periodic boundary conditions:
            L(u, v) = 2·[cos(2π u / W) - 1] + 2·[cos(2π v / H) - 1]
        Regularized transfer function:
            H_inv(u, v) = L(u, v) / [ L(u, v)² + λ_reg ]
        """
        if self._laplacian_kernel_dft is not None and self._dft_h == h and self._dft_w == w:
            return

        self._dft_h = h
        self._dft_w = w

        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u, v)

        # Exact 5-point discrete Laplacian eigenvalues in frequency domain
        laplacian_eigenvalues = 2.0 * (np.cos(2.0 * np.pi * u_grid / w) - 1.0) + \
                                2.0 * (np.cos(2.0 * np.pi * v_grid / h) - 1.0)

        # Regularized pseudo-inverse with zero DC component (preserves gauge symmetry)
        reg = self.config.poisson_regularization
        denom = laplacian_eigenvalues**2 + reg
        inv_kernel = laplacian_eigenvalues / denom
        inv_kernel[0, 0] = 0.0  # Singularity removal at DC (k_x = 0, k_y = 0)

        self._laplacian_kernel_dft = inv_kernel.astype(np.float32)

    def solve_poisson_2d(self, divergence_field: np.ndarray) -> np.ndarray:
        """
        Solves ∇² Φ = div(d) / C_BOS using fast 2D Discrete Fourier Transforms.

        Returns:
            phi: 2D reconstructed relative optical path difference (OPD) field (meters).
                 Directly proportional to the line-of-sight integrated gas density:
                 Φ(x, y) = ∫ [n(x, y, z) - n_0] dz = K_GD · ∫ [ρ(x, y, z) - ρ_0] dz
        """
        h, w = divergence_field.shape
        if self.config.use_optimal_dft_padding:
            opt_h = cv2.getOptimalDFTSize(h)
            opt_w = cv2.getOptimalDFTSize(w)
            if opt_h != h or opt_w != w:
                padded_div = cv2.copyMakeBorder(
                    divergence_field, 0, opt_h - h, 0, opt_w - w,
                    cv2.BORDER_REFLECT_101
                )
            else:
                padded_div = divergence_field
        else:
            opt_h, opt_w = h, w
            padded_div = divergence_field

        self._precompute_poisson_kernel(opt_h, opt_w)

        # Scale divergence by physical BOS coupling constant C_BOS
        c_bos = self.config.geometry.bos_coupling_coefficient
        scaled_div = padded_div / (c_bos + 1e-12)

        # Forward 2D Real-to-Complex FFT (cv2.dft)
        dft_div = cv2.dft(scaled_div.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

        # Frequency domain multiplication: F{Φ} = F{div} * H_inv
        # cv2.dft returns (H, W, 2) where channel 0 is Real, channel 1 is Imag
        kernel = self._laplacian_kernel_dft
        dft_div[:, :, 0] *= kernel
        dft_div[:, :, 1] *= kernel

        # Inverse 2D Complex-to-Real FFT (cv2.idft)
        idft_result = cv2.idft(dft_div, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Crop back to unpadded frame dimensions
        phi_reconstructed = idft_result[:h, :w]
        return phi_reconstructed

    # =========================================================================
    # Mathematical Physics: Temporal Bandpass Filtering
    # =========================================================================
    def _setup_temporal_bandpass(self) -> None:
        """
        Designs a 2nd-order digital Butterworth bandpass filter in Direct Form II.
        Separates high-frequency camera noise from low-frequency wind drift:
            H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        """
        fs = self.config.fps
        f_low = self.config.bandpass_low_hz
        f_high = self.config.bandpass_high_hz

        # Angular cutoff frequencies (pre-warped for bilinear transform)
        w_low = 2.0 * math.pi * f_low / fs
        w_high = 2.0 * math.pi * f_high / fs
        w0 = math.sqrt(w_low * w_high)
        bw = w_high - w_low

        q = w0 / bw
        alpha = math.sin(w0) / (2.0 * q)

        # Biquad bandpass coefficients (constant 0 dB peak gain)
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * math.cos(w0)
        a2 = 1.0 - alpha

        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0
        self._biquad_init = True

    def _apply_temporal_bandpass(self, field_2d: np.ndarray) -> np.ndarray:
        """
        Applies Direct Form II transposed biquad temporal filtering to every pixel in the 2D field.
        Maintains minimal state matrices w1 and w2 for zero allocation overhead.
        """
        if self._w1 is None or self._w1.shape != field_2d.shape:
            self._w1 = np.zeros_like(field_2d, dtype=np.float32)
            self._w2 = np.zeros_like(field_2d, dtype=np.float32)

        x = field_2d.astype(np.float32)
        # Direct Form II:
        # y[n] = b0 * x[n] + w1[n-1]
        # w1[n] = b1 * x[n] - a1 * y[n] + w2[n-1]
        # w2[n] = b2 * x[n] - a2 * y[n]
        y = self._b0 * x + self._w1
        self._w1 = self._b1 * x - self._a1 * y + self._w2
        self._w2 = self._b2 * x - self._a2 * y
        return y

    # =========================================================================
    # High-Performance Processing Pipeline
    # =========================================================================
    def process_frame(
        self,
        bgr_frame: np.ndarray,
        anomaly_feedback_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single video frame through the complete optical Schlieren pipeline.

        Args:
            bgr_frame: Input color frame (uint8, BGR or Grayscale) at native resolution.
            anomaly_feedback_mask: Optional boolean/uint8 mask of detected plume from previous frame
                                   to freeze background adaptation.

        Returns:
            Tuple of:
                flow_field: (H, W, 2) sub-pixel refractive displacement vectors (u, v) in pixels.
                divergence: (H, W) spatial divergence of displacement field (∇ · d).
                reconstructed_density: (H, W) 2D relative optical path difference field Φ (meters).
                turbulent_bandpassed: (H, W) temporal bandpass filtered gas concentration anomaly.
        """
        t_start = time.perf_counter()

        # Step 1: Preprocessing & Scale Handling
        t0 = time.perf_counter()
        if len(bgr_frame.shape) == 3:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_frame

        scale = self.config.processing_scale
        if scale != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        norm_current = self._normalize_illumination(gray)

        if self._bg_reference is None:
            self._update_background(norm_current, None)
            h, w = gray.shape
            zero_flow = np.zeros((h, w, 2), dtype=np.float32)
            zero_field = np.zeros((h, w), dtype=np.float32)
            return zero_flow, zero_field, zero_field, zero_field

        t_preprocess = (time.perf_counter() - t0) * 1000.0

        # Step 2: Dense Sub-Pixel Optical Flow via OpenCV 5 DIS
        t0 = time.perf_counter()
        bg_uint8 = np.clip(self._bg_reference, 0, 255).astype(np.uint8)
        
        # DIS optical flow: measures displacement from reference background to current frame
        flow = self._dis_flow.calc(bg_uint8, norm_current, None)
        t_flow = (time.perf_counter() - t0) * 1000.0

        # Step 3: Divergence Field Calculation
        # ∇ · d = ∂u/∂x + ∂v/∂y
        # Computed using central finite differences via cv2.Sobel or cv2.Scharr
        t0 = time.perf_counter()
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        
        # Scaled central difference: ∂u/∂x (Scharr or Sobel with dx=1, dy=0)
        du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
        dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
        divergence = du_dx + dv_dy
        t_div = (time.perf_counter() - t0) * 1000.0

        # Step 4: 2D Poisson Reconstruction (Fourier Inversion)
        t0 = time.perf_counter()
        phi_reconstructed = self.solve_poisson_2d(divergence)
        t_poisson = (time.perf_counter() - t0) * 1000.0

        # Step 5: Temporal Bandpass Filtering (Turbulent Eddy Isolation)
        t0 = time.perf_counter()
        turbulent_anomaly = self._apply_temporal_bandpass(phi_reconstructed)
        t_filter = (time.perf_counter() - t0) * 1000.0

        # Step 6: Background Reference Adaptation
        self._update_background(norm_current, anomaly_feedback_mask)
        self._frame_count += 1

        t_total = (time.perf_counter() - t_start) * 1000.0
        self.last_profile_ms = {
            "preprocess": t_preprocess,
            "optical_flow": t_flow,
            "divergence": t_div,
            "poisson_solve": t_poisson,
            "temporal_filter": t_filter,
            "total": t_total,
        }

        return flow, divergence, phi_reconstructed, turbulent_anomaly

    def reset_background(self) -> None:
        """Resets the continuous background reference model."""
        self._bg_reference = None
        self._bg_variance = None
        self._frame_count = 0
        self._w1 = None
        self._w2 = None
