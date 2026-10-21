"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: process_video.py

Standalone video processing pipeline:
- Ingests video files (.mp4, .avi, .mov).
- Computes dense sub-pixel optical flow using OpenCV DISOpticalFlow.
- Evaluates spatial divergence (du/dx + dv/dy) to isolate fluid expansion from camera drift.
- Segments anomalous refractive plumes via dynamic statistical thresholding.
- Applies colormapped refractive displacement overlay with alpha blending.
- Draws plume contours, bounding boxes, and burns frame telemetry metrics.
- Outputs annotated H.264 / MP4 video using cv2.VideoWriter and ffmpeg.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def normalize_illumination(gray_frame: np.ndarray, ksize: int = 31) -> np.ndarray:
    """
    Applies local contrast normalization to isolate surface texture from
    spatially varying illumination and lighting drift.
    """
    if ksize % 2 == 0:
        ksize += 1
    img_f = gray_frame.astype(np.float32)
    local_mean = cv2.boxFilter(img_f, cv2.CV_32F, (ksize, ksize), normalize=True)
    local_sq_mean = cv2.boxFilter(img_f * img_f, cv2.CV_32F, (ksize, ksize), normalize=True)
    local_var = np.maximum(local_sq_mean - (local_mean * local_mean), 1.0)
    local_std = np.sqrt(local_var)
    norm = (img_f - local_mean) / (local_std + 1e-3)
    norm_uint8 = np.clip(norm * 42.0 + 128.0, 0, 255).astype(np.uint8)
    return norm_uint8


def process_video_file(
    input_path: str,
    output_path: str,
    preset: str = "medium",
    threshold_sigma: float = 2.5,
    min_area_px: int = 150,
    colormap_name: str = "JET",
    alpha_blend: float = 0.65,
    reference_mode: str = "static",
    scale: float = 1.0,
    max_frames: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, any]:
    """
    Processes an input video file using genuine OpenCV DIS Optical Flow and
    Background-Oriented Schlieren divergence analysis.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file with OpenCV: {input_path}")

    # Video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Output dimensions
    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)
    out_w = out_w - (out_w % 2)  # Ensure even dimensions for video codecs
    out_h = out_h - (out_h % 2)

    # Configure DIS Optical Flow preset
    dis_presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    dis_flag = dis_presets.get(preset.lower(), cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis = cv2.DISOpticalFlow_create(dis_flag)

    # Configure Colormap
    colormaps = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "HOT": cv2.COLORMAP_HOT,
        "MAGMA": cv2.COLORMAP_MAGMA,
    }
    cv_colormap = colormaps.get(colormap_name.upper(), cv2.COLORMAP_JET)

    # Intermediate output file for OpenCV VideoWriter
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_output_path = output_path.replace(".mp4", "_temp_raw.mp4")

    # Initialize VideoWriter (mp4v is broadly supported on Windows OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
    if not out_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))

    # Read reference frame (Frame 0)
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Failed to read the first frame of the video.")

    if scale != 1.0:
        first_frame = cv2.resize(first_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    reference_gray = normalize_illumination(first_gray)

    # Initial frame output (write first frame with initial HUD)
    hud_first = first_frame.copy()
    cv2.rectangle(hud_first, (0, 0), (out_w, 40), (15, 23, 42), -1)
    cv2.putText(hud_first, "AEROBOS: INITIALIZING REFERENCE BASELINE (FRAME 1)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    out_writer.write(hud_first)

    # Statistics tracking
    frame_idx = 1
    plume_detections = 0
    max_displacement = 0.0
    total_area_accum = 0
    t_start = time.perf_counter()
    prev_gray = reference_gray.copy()

    print(f"[*] Processing '{input_path}': {total_frames} frames ({out_w}x{out_h} @ {fps:.1f} FPS)...")

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        t_frame_start = time.perf_counter()

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        current_color = frame.copy()
        current_gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = normalize_illumination(current_gray_raw)

        # 1. Optical Flow Calculation
        if reference_mode == "running":
            # Compare with previous frame (sensitive to high-frequency temporal turbulence)
            flow = dis.calc(prev_gray, current_gray, None)
            prev_gray = current_gray.copy()
        else:
            # Compare with static reference background (measures absolute displacement from resting baseline)
            flow = dis.calc(reference_gray, current_gray, None)

        u = flow[..., 0]
        v = flow[..., 1]

        # 2. Refraction / Plume Extraction
        # Displacement Magnitude & Angle
        mag, ang = cv2.cartToPolar(u, v)
        mean_disp = float(np.mean(mag))
        max_disp = float(np.max(mag))
        if max_disp > max_displacement:
            max_displacement = max_disp

        # Divergence Calculation: du/dx + dv/dy using central Sobel operators
        du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
        dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
        divergence = du_dx + dv_dy
        abs_div = np.abs(divergence)

        # Refractive Metric: combine divergence (dilation/compression) with displacement magnitude
        refractive_metric = abs_div * (mag + 0.1)

        # Dynamic Thresholding: mean + (threshold_sigma * std)
        metric_mean = float(np.mean(refractive_metric))
        metric_std = float(np.std(refractive_metric))
        thresh_val = metric_mean + (threshold_sigma * metric_std)

        # Binary Anomaly Mask
        raw_mask = (refractive_metric > thresh_val).astype(np.uint8) * 255

        # Morphological Cleanup (bridge turbulent eddies and suppress single-pixel shot noise)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # Find Connected Contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]

        # 3. Visualization Overlay
        # Normalize magnitude to 0-255 for colormap
        norm_mag = np.clip((mag / (np.percentile(mag, 99.0) + 1e-5)) * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_mag, cv_colormap)

        # Alpha-blend colormap over original frame
        mask_3ch = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR) / 255.0
        blended = (current_color * (1.0 - mask_3ch * alpha_blend) + heatmap * (mask_3ch * alpha_blend)).astype(np.uint8)

        total_plume_pixels = 0
        is_anomaly = len(valid_contours) > 0

        if is_anomaly:
            plume_detections += 1
            for cnt in valid_contours:
                area = int(cv2.contourArea(cnt))
                total_plume_pixels += area
                total_area_accum += area

                # Draw contour polyline (Red)
                cv2.polylines(blended, [cnt], isClosed=True, color=(0, 0, 255), thickness=2)

                # Draw bounding box (Orange)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)

            # Mark dominant centroid
            dominant_cnt = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(dominant_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(blended, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawMarker(blended, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

        # Draw Decimated Quiver Vectors in active flow areas
        step = max(24, int(out_w / 30))
        y_pts, x_pts = np.mgrid[step // 2:out_h:step, step // 2:out_w:step]
        u_sub = u[y_pts, x_pts]
        v_sub = v[y_pts, x_pts]
        mag_sub = np.sqrt(u_sub**2 + v_sub**2)
        for y_p, x_p, u_p, v_p, m_p in zip(y_pts.ravel(), x_pts.ravel(), u_sub.ravel(), v_sub.ravel(), mag_sub.ravel()):
            if m_p > 0.20:
                scale_arr = 5.0
                pt1 = (int(x_p), int(y_p))
                pt2 = (int(x_p + u_p * scale_arr), int(y_p + v_p * scale_arr))
                cv2.arrowedLine(blended, pt1, pt2, (0, 255, 255), 1, tipLength=0.3)

        # Frame Processing Latency
        t_frame_end = time.perf_counter()
        frame_fps = 1.0 / max(t_frame_end - t_frame_start, 1e-4)

        # 4. Burn Frame Telemetry HUD
        # Top HUD Banner
        cv2.rectangle(blended, (0, 0), (out_w, 54), (15, 23, 42), -1)
        banner_border = (0, 0, 255) if is_anomaly else (34, 197, 94)
        cv2.line(blended, (0, 54), (out_w, 54), banner_border, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Left Title
        cv2.putText(blended, "AEROBOS: OPTICAL GAS IMAGING PIPELINE", (14, 22), font, 0.55, (255, 255, 255), 2)
        cv2.putText(
            blended,
            f"FRAME: {frame_idx:04d}/{total_frames:04d} | PRESET: {preset.upper()} | RES: {out_w}x{out_h}",
            (14, 44),
            font,
            0.40,
            (148, 163, 184),
            1,
        )

        # Right Telemetry Metrics
        status_text = "FUGITIVE GAS PLUME" if is_anomaly else "BASELINE CLEAR"
        status_col = (0, 0, 255) if is_anomaly else (34, 197, 94)
        cv2.putText(blended, status_text, (out_w - 230, 22), font, 0.52, status_col, 2)
        metrics_str = f"DISP: {mean_disp:.2f}px | AREA: {total_plume_pixels}px | {frame_fps:.1f} FPS"
        cv2.putText(blended, metrics_str, (out_w - 250, 44), font, 0.38, (255, 255, 255), 1)

        # Bottom SCADA Status
        cv2.rectangle(blended, (0, out_h - 26), (out_w, out_h), (15, 23, 42), -1)
        if is_anomaly:
            scada_msg = f"SCADA INTERLOCK: PLUME IDENTIFIED ({total_plume_pixels} PIXELS) | TRIP STANDBY"
            scada_col = (0, 69, 255)
        else:
            scada_msg = "SCADA SUPERVISORY: NORMAL PROCESS BASELINE"
            scada_col = (34, 197, 94)
        cv2.putText(blended, scada_msg, (14, out_h - 8), font, 0.38, scada_col, 1)

        out_writer.write(blended)

        if progress_callback:
            progress_callback(frame_idx / float(max(total_frames, 1)))

    cap.release()
    out_writer.release()

    total_proc_time = time.perf_counter() - t_start
    overall_fps = frame_idx / max(total_proc_time, 1e-4)

    # Convert temp video to web-standard H.264 using ffmpeg if available
    final_output_path = output_path
    if shutil.which("ffmpeg"):
        print(f"[*] Re-encoding to web-compatible H.264 using ffmpeg -> {output_path}...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "main",
            "-level",
            "3.1",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            output_path,
        ]
        res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        else:
            # Fallback to temp file if ffmpeg encountered an issue
            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, output_path)
    else:
        if os.path.exists(temp_output_path):
            shutil.move(temp_output_path, output_path)

    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "total_frames_processed": frame_idx,
        "plume_frames_detected": plume_detections,
        "detection_rate_pct": (plume_detections / max(frame_idx, 1)) * 100.0,
        "max_displacement_px": max_displacement,
        "avg_plume_area_px": total_area_accum / max(plume_detections, 1),
        "total_processing_time_s": total_proc_time,
        "overall_fps": overall_fps,
        "resolution": f"{out_w}x{out_h}",
    }

    print(f"[OK] Video processing complete! Output saved to: {output_path}")
    print(f"     Processed {frame_idx} frames in {total_proc_time:.2f}s ({overall_fps:.1f} FPS)")
    print(f"     Plume detected in {plume_detections}/{frame_idx} frames ({stats['detection_rate_pct']:.1f}%)")
    return stats


def main() -> None:
    """CLI entry point for process_video.py."""
    parser = argparse.ArgumentParser(
        description="AeroBOS: Real-Time Software-Defined Optical Gas Imaging Video Processor"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4, .avi, .mov)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output annotated video (.mp4)")
    parser.add_argument("--preset", type=str, default="medium", choices=["ultrafast", "fast", "medium"], help="DIS flow preset")
    parser.add_argument("--threshold-sigma", type=float, default=2.5, help="Statistical anomaly threshold multiplier")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum plume pixel area")
    parser.add_argument("--colormap", type=str, default="JET", choices=["JET", "TURBO", "VIRIDIS", "HOT"], help="Heatmap colormap")
    parser.add_argument("--alpha", type=float, default=0.65, help="Heatmap alpha blend opacity (0.0 - 1.0)")
    parser.add_argument("--reference-mode", type=str, default="static", choices=["static", "running"], help="Reference background mode")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scaling factor")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")

    args = parser.parse_args()

    try:
        process_video_file(
            input_path=args.input,
            output_path=args.output,
            preset=args.preset,
            threshold_sigma=args.threshold_sigma,
            min_area_px=args.min_area,
            colormap_name=args.colormap,
            alpha_blend=args.alpha,
            reference_mode=args.reference_mode,
            scale=args.scale,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
