"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: plume_analysis.py

Implements PlumeAnalyzer:
- Dynamic 3-sigma statistical noise floor estimation.
- Morphological segmentation and contour extraction of turbulent refractive plumes.
- Quantitative tracking: centroid, bounding box, spatial area (m²), boundary expansion rate (dA/dt),
  dispersion orientation angle, and maximum line-of-sight optical path difference (OPD).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .refractometer import OpticalGeometry


@dataclass
class PlumeMetrics:
    """Quantitative morphological and physical metrics of a detected gas plume."""
    is_anomaly_detected: bool
    snr_sigma: float
    confidence_score: float  # [0.0, 1.0]
    
    # Spatial coordinates (image plane)
    centroid_px: Tuple[float, float]
    centroid_normalized: Tuple[float, float]
    bounding_box_xywh: Tuple[int, int, int, int]
    contour_points: List[Tuple[int, int]] = field(default_factory=list)
    
    # Physical dimensions in real-world metric units
    area_pixels: int = 0
    area_sq_meters: float = 0.0
    expansion_rate_sq_m_per_sec: float = 0.0  # dA/dt
    dispersion_angle_deg: float = 0.0
    
    # Optical metrics
    peak_optical_path_diff_nm: float = 0.0    # Peak OPD in nanometers
    mean_divergence_anomaly: float = 0.0
    timestamp_epoch: float = field(default_factory=time.time)


class PlumeAnalyzer:
    """
    Analyzes turbulent refractive field fluctuations to detect, segment,
    and quantitatively measure gas leak plumes.
    """

    def __init__(
        self,
        geometry: Optional[OpticalGeometry] = None,
        sigma_threshold: float = 3.5,
        min_plume_area_pixels: int = 400,
        history_window_frames: int = 60,
    ) -> None:
        self.geometry = geometry or OpticalGeometry()
        self.sigma_threshold = sigma_threshold
        self.min_plume_area_pixels = min_plume_area_pixels
        self.history_window = history_window_frames

        # Online running noise floor statistics (Welford algorithm)
        self._noise_mean: float = 1e-5
        self._noise_var: float = 1e-10
        self._noise_sample_count: int = 0

        # Temporal expansion tracking
        self._last_area_sq_m: float = 0.0
        self._last_detection_timestamp: float = 0.0

    @property
    def meters_per_pixel(self) -> float:
        """
        Metric resolution at the gas disturbance plane Z_D:
            s_gas = (Z_D / f) * p_pix (meters per pixel)
        """
        return (self.geometry.dist_camera_to_gas_m / self.geometry.focal_length_m) * self.geometry.pixel_pitch_m

    def update_noise_floor(self, metric_sample: float) -> None:
        """Updates background noise mean and variance using Welford's algorithm."""
        self._noise_sample_count += 1
        delta = metric_sample - self._noise_mean
        self._noise_mean += delta / min(self._noise_sample_count, self.history_window)
        delta2 = metric_sample - self._noise_mean
        self._noise_var += (delta * delta2 - self._noise_var) / min(self._noise_sample_count, self.history_window)

    def analyze_field(
        self,
        turbulent_opd_field: np.ndarray,
        divergence_field: np.ndarray,
    ) -> Tuple[PlumeMetrics, np.ndarray]:
        """
        Extracts plume metrics and binary segmentation mask from the reconstructed refractive field.

        Args:
            turbulent_opd_field: (H, W) bandpass-filtered optical path difference (meters).
            divergence_field: (H, W) spatial divergence field.

        Returns:
            Tuple of (PlumeMetrics, binary_mask_uint8)
        """
        current_time = time.time()
        abs_opd = np.abs(turbulent_opd_field)
        
        # Current frame spatial median and robust MAD (Median Absolute Deviation)
        frame_median = float(np.median(abs_opd))
        frame_mad = float(np.median(np.abs(abs_opd - frame_median)))
        frame_noise_std = max(1.4826 * frame_mad, 1e-9)

        # Update noise floor baseline
        self.update_noise_floor(frame_noise_std)
        effective_noise_std = max(self._noise_mean, frame_noise_std)
        
        # 3-Sigma thresholding based on robust spatial noise floor
        threshold_val = frame_median + self.sigma_threshold * effective_noise_std
        raw_mask = (abs_opd > threshold_val).astype(np.uint8) * 255

        # Morphological filtering: remove isolated noise specks, bridge plume eddies
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = turbulent_opd_field.shape
        meters_per_px = self.meters_per_pixel
        area_scale_factor = meters_per_px * meters_per_px

        # Warmup guard: suppress triggers during the first 5 calibration frames
        is_warming_up = self._noise_sample_count <= 5

        # Filter candidate contours by area threshold
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_plume_area_pixels]

        if not valid_contours or is_warming_up:
            # No significant anomaly
            metrics = PlumeMetrics(
                is_anomaly_detected=False,
                snr_sigma=float(frame_median / (effective_noise_std + 1e-12)),
                confidence_score=0.0,
                centroid_px=(w / 2.0, h / 2.0),
                centroid_normalized=(0.5, 0.5),
                bounding_box_xywh=(0, 0, 0, 0),
                timestamp_epoch=current_time,
            )
            return metrics, cleaned_mask

        # Select the dominant plume contour (largest area)
        dominant_contour = max(valid_contours, key=cv2.contourArea)
        area_px = int(cv2.contourArea(dominant_contour))
        area_sq_m = float(area_px * area_scale_factor)

        # Compute centroid via spatial image moments
        M = cv2.moments(dominant_contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(w / 2.0), float(h / 2.0)

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(dominant_contour)

        # Dispersion angle from orientation ellipse
        dispersion_angle = 0.0
        if len(dominant_contour) >= 5:
            ellipse = cv2.fitEllipse(dominant_contour)
            dispersion_angle = float(ellipse[2])

        # Boundary expansion rate (dA/dt in m²/s)
        expansion_rate = 0.0
        if self._last_detection_timestamp > 0 and (current_time - self._last_detection_timestamp) > 0:
            dt = current_time - self._last_detection_timestamp
            expansion_rate = max(0.0, (area_sq_m - self._last_area_sq_m) / dt)

        self._last_area_sq_m = area_sq_m
        self._last_detection_timestamp = current_time

        # Peak Optical Path Difference in nanometers inside the plume
        plume_mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(plume_mask_roi, [dominant_contour], -1, 255, -1)
        peak_opd_meters = float(np.max(abs_opd[plume_mask_roi > 0]))
        peak_opd_nm = peak_opd_meters * 1e9  # Convert meters to nanometers

        # Plume SNR over baseline noise
        snr_sigma = float((peak_opd_meters - self._noise_mean) / (effective_noise_std + 1e-12))
        confidence = float(np.clip((snr_sigma - self.sigma_threshold) / 5.0, 0.0, 1.0))
        is_anomaly = (snr_sigma >= self.sigma_threshold) and (confidence > 0.05)

        # Mean divergence within plume core
        mean_div = float(np.mean(divergence_field[plume_mask_roi > 0]))

        # Contour approximation for JSON/MQTT lightweight serialization
        approx_contour = cv2.approxPolyDP(dominant_contour, epsilon=2.0, closed=True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx_contour]

        metrics = PlumeMetrics(
            is_anomaly_detected=is_anomaly,
            snr_sigma=snr_sigma,
            confidence_score=confidence,
            centroid_px=(cx, cy),
            centroid_normalized=(cx / float(w), cy / float(h)),
            bounding_box_xywh=(bx, by, bw, bh),
            contour_points=pts,
            area_pixels=area_px,
            area_sq_meters=area_sq_m,
            expansion_rate_sq_m_per_sec=expansion_rate,
            dispersion_angle_deg=dispersion_angle,
            peak_optical_path_diff_nm=peak_opd_nm,
            mean_divergence_anomaly=mean_div,
            timestamp_epoch=current_time,
        )

        return metrics, plume_mask_roi
