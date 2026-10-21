"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging Video Processor
Module: apps/process_video.py

CLI Video Processing Pipeline:
- Ingests video via cv2.VideoCapture.
- Computes dense sub-pixel optical flow using BOSCore (DISOpticalFlow).
- Inverts 2D Poisson equation via FFT (solve_poisson_fft).
- Segments plume and renders COLORMAP_TURBO heatmap and bounding boxes via PlumeAnalyzer.
- Writes annotated video to --output via cv2.VideoWriter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# Ensure parent directory is in path for aerobos import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aerobos.bos_core import BOSCore
from aerobos.plume_metrics import PlumeAnalyzer


def process_video(
    input_path: str,
    output_path: str,
    sigma_threshold: float = 2.5,
    min_area_px: int = 150,
    reference_mode: str = "static",
    scale: float = 1.0,
) -> None:
    """
    Processes every frame of a video using BOSCore and PlumeAnalyzer,
    writing the annotated output to output_path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")

    # Video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)
    out_w -= out_w % 2
    out_h -= out_h % 2

    # Instantiate Core Physics & Metrics
    bos = BOSCore(preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    analyzer = PlumeAnalyzer(sigma_threshold=sigma_threshold, min_contour_area=min_area_px)

    # Initialize VideoWriter
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (out_w, out_h))

    # Read first frame as reference baseline
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Failed to read the initial frame from video.")

    if scale != 1.0:
        first_frame = cv2.resize(first_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    ref_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray.copy()

    # Initial frame output
    hud_first = first_frame.copy()
    cv2.rectangle(hud_first, (0, 0), (out_w, 40), (15, 23, 42), -1)
    cv2.putText(
        hud_first,
        "AEROBOS: INITIALIZING BASELINE (FRAME 1)",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    writer.write(hud_first)

    frame_idx = 1
    detected_count = 0
    t_start = time.perf_counter()

    print(f"[*] AeroBOS Processing: '{input_path}' ({total_frames} frames, {out_w}x{out_h} @ {fps:.1f} FPS)...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        t_frame_start = time.perf_counter()

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Optical Flow Computation
        if reference_mode == "running":
            flow = bos.calc_flow(prev_gray, curr_gray)
            prev_gray = curr_gray.copy()
        else:
            flow = bos.calc_flow(ref_gray, curr_gray)

        # 2. Spatial Divergence
        div = bos.compute_divergence(flow)

        # 3. 2D Poisson FFT Inversion to Reconstruct OPD
        opd = bos.solve_poisson_fft(div)

        # 4. Plume Segmentation, Bounding Box, and TURBO Overlay
        annotated_frame, metrics = analyzer.analyze(opd, frame)

        if metrics["is_anomaly_detected"]:
            detected_count += 1

        t_frame_end = time.perf_counter()
        instant_fps = 1.0 / max(t_frame_end - t_frame_start, 1e-4)

        # Additional frame watermark
        cv2.putText(
            annotated_frame,
            f"FRAME {frame_idx}/{total_frames} | {instant_fps:.1f} FPS",
            (out_w - 240, out_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

        writer.write(annotated_frame)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"    Frame [{frame_idx:04d}/{total_frames:04d}] - Plume: {metrics['is_anomaly_detected']} - Area: {metrics['total_plume_area_pixels']} px - Speed: {instant_fps:.1f} FPS")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / max(total_time, 1e-4)

    # Re-encode to universal H.264 MP4 if ffmpeg is available
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", temp_output_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "main", "-level", "3.1",
            "-crf", "20", "-movflags", "+faststart",
            output_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        else:
            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, output_path)
    else:
        if os.path.exists(temp_output_path):
            shutil.move(temp_output_path, output_path)

    print(f"[OK] Successfully processed {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS).")
    print(f"     Plume detected in {detected_count}/{frame_idx} frames ({(detected_count/frame_idx)*100:.1f}%).")
    print(f"     Annotated output written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroBOS: Real-Time Optical Gas Imaging Video Processor")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4, .avi, .mov)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output annotated video (.mp4)")
    parser.add_argument("--sigma", type=float, default=2.5, help="Adaptive sigma threshold multiplier")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum plume contour area in pixels")
    parser.add_argument("--reference-mode", type=str, default="static", choices=["static", "running"], help="Reference mode")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scaling factor")

    args = parser.parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        sigma_threshold=args.sigma,
        min_area_px=args.min_area,
        reference_mode=args.reference_mode,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
