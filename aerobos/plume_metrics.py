"""
AeroBOS: Plume Analysis and Optical Path Difference Overlay
Module: aerobos/plume_metrics.py

Implements PlumeAnalyzer:
- Adaptive sigma thresholding on reconstructed OPD fields.
- Morphological noise suppression and edge bridging via cv2.morphologyEx.
- Connected component contour extraction via cv2.findContours.
- Augmented reality overlay: bounding boxes and cv2.COLORMAP_TURBO heatmap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class PlumeAnalyzer:
    """
    Analyzes reconstructed Optical Path Difference (OPD) fields to detect,
    segment, and visually annotate anomalous gas plumes.
    """

    def __init__(
        self,
        sigma_threshold: float = 2.5,
        min_contour_area: int = 150,
        alpha_blend: float = 0.65,
    ) -> None:
        """
        Args:
            sigma_threshold: Multiplier for adaptive threshold (mean + sigma * std).
            min_contour_area: Minimum connected contour area in pixels to retain.
            alpha_blend: Opacity factor for blending the TURBO heatmap over the frame.
        """
        self.sigma_threshold = sigma_threshold
        self.min_contour_area = min_contour_area
        self.alpha_blend = alpha_blend

    def analyze(
        self,
        opd: np.ndarray,
        original_frame: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies adaptive thresholding, morphological filtering, and contour extraction
        to the OPD field, rendering an augmented overlay with bounding boxes and
        a COLORMAP_TURBO heatmap.

        Args:
            opd: (H, W) reconstructed scalar Optical Path Difference field.
            original_frame: (H, W, 3) original color video frame (BGR).

        Returns:
            Tuple of:
                annotated_frame: (H, W, 3) BGR image with blended TURBO heatmap and bounding boxes.
                metrics: Dictionary containing anomaly flag, plume area, and contour statistics.
        """
        h, w = original_frame.shape[:2]
        abs_opd = np.abs(opd)

        # 1. Adaptive Sigma Thresholding
        mean_val = float(np.mean(abs_opd))
        std_val = float(np.std(abs_opd))
        threshold_val = mean_val + (self.sigma_threshold * std_val)

        raw_mask = (abs_opd > threshold_val).astype(np.uint8) * 255

        # 2. Morphological Operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 3. Contour Extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours: List[np.ndarray] = [
            c for c in contours if cv2.contourArea(c) >= self.min_contour_area
        ]

        # 4. COLORMAP_TURBO Heatmap Generation
        p99 = np.percentile(abs_opd, 99.0) + 1e-6
        norm_opd = np.clip((abs_opd / p99) * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_opd, cv2.COLORMAP_TURBO)

        # 5. Alpha-Blend Heatmap onto Original Frame within Plume Mask
        mask_3ch = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR) / 255.0
        annotated_frame = (
            original_frame * (1.0 - mask_3ch * self.alpha_blend)
            + heatmap * (mask_3ch * self.alpha_blend)
        ).astype(np.uint8)

        total_plume_pixels = 0
        bounding_boxes = []

        # 6. Render Bounding Boxes and Contours
        for cnt in valid_contours:
            area = int(cv2.contourArea(cnt))
            total_plume_pixels += area

            # Draw contour boundary (Red)
            cv2.polylines(annotated_frame, [cnt], isClosed=True, color=(0, 0, 255), thickness=2)

            # Draw bounding box (Orange)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            bounding_boxes.append((bx, by, bw, bh))
            cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)

        is_anomaly = len(valid_contours) > 0

        # Burn live telemetry metrics onto the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(annotated_frame, (0, 0), (w, 40), (15, 23, 42), -1)
        border_col = (0, 0, 255) if is_anomaly else (34, 197, 94)
        cv2.line(annotated_frame, (0, 40), (w, 40), border_col, 2)

        status_str = "ANOMALY: FUGITIVE PLUME DETECTED" if is_anomaly else "STATUS: BASELINE NOMINAL"
        status_col = (0, 0, 255) if is_anomaly else (34, 197, 94)
        cv2.putText(annotated_frame, status_str, (14, 26), font, 0.55, status_col, 2)

        stats_str = f"PLUME AREA: {total_plume_pixels} px | MEAN OPD: {mean_val:.4f}"
        cv2.putText(annotated_frame, stats_str, (w - 380, 26), font, 0.48, (255, 255, 255), 1)

        metrics = {
            "is_anomaly_detected": is_anomaly,
            "total_plume_area_pixels": total_plume_pixels,
            "num_plumes": len(valid_contours),
            "bounding_boxes": bounding_boxes,
            "mean_opd": mean_val,
            "std_opd": std_val,
            "max_opd": float(np.max(abs_opd)),
        }

        return annotated_frame, metrics
