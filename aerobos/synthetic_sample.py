"""
AeroBOS: Synthetic Physical Test Video Generator
Module: aerobos/synthetic_sample.py

Generates a 5-second test video (sample_test_video.mp4) at 30 FPS:
- High-contrast gravel background texture generated using np.random.randint.
- Frames 0 to 30: Baseline static background without gas disturbance.
- After frame 30: Moving 2D Gaussian refractive spatial displacement applied via cv2.remap
  to physically simulate light ray bending through an expanding turbulent gas plume.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import cv2
import numpy as np


def generate_sample_video(
    output_path: str = "sample_test_video.mp4",
    width: int = 640,
    height: int = 360,
    fps: float = 30.0,
    duration_sec: float = 5.0,
) -> str:
    """
    Generates a 5-second synthetic test video with a high-contrast gravel texture
    and moving 2D Gaussian refractive displacement.
    """
    total_frames = int(fps * duration_sec)  # 150 frames
    temp_path = output_path.replace(".mp4", "_raw.mp4")

    # 1. High-Contrast Gravel Background Texture using np.random.randint
    np.random.seed(42)
    fine_gravel = np.random.randint(30, 230, size=(height, width, 3), dtype=np.uint8)
    coarse_pebbles = np.random.randint(40, 210, size=(height // 6, width // 6, 3), dtype=np.uint8)
    coarse_resized = cv2.resize(coarse_pebbles, (width, height), interpolation=cv2.INTER_NEAREST)
    base_gravel = cv2.addWeighted(fine_gravel, 0.65, coarse_resized, 0.35, 0)

    # 2. Setup OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    print(f"[*] Generating {duration_sec}s test video ({total_frames} frames @ {fps} FPS) -> {output_path}...")

    # Coordinate grid for spatial remap
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    for frame_idx in range(total_frames):
        if frame_idx <= 30:
            # Baseline period (Frames 0 to 30): Static gravel scene with slight sensor shot noise
            noise = np.random.randint(-2, 3, size=base_gravel.shape, dtype=np.int16)
            frame = np.clip(base_gravel.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # After Frame 30: Moving 2D Gaussian refractive spatial displacement
            t = (frame_idx - 30) / fps  # Time in seconds (0.0 to 4.0s)

            # Plume centroid moves upward and drifts rightward
            center_x = (width * 0.40) + (45.0 * t)
            center_y = (height * 0.80) - (55.0 * t)

            # Expanding dispersion envelope
            sigma_x = 25.0 + (12.0 * math.sqrt(max(t, 0.01)))
            sigma_y = 35.0 + (18.0 * math.sqrt(max(t, 0.01)))

            # Gaussian refractive density perturbation
            dist_sq = ((xx - center_x) ** 2) / (2.0 * sigma_x**2) + ((yy - center_y) ** 2) / (2.0 * sigma_y**2)
            amplitude = 4.2 * (1.0 + 0.20 * math.sin(t * 12.0))  # Turbulent oscillation
            phi = np.exp(-dist_sq) * amplitude

            # Snell's Law spatial ray deflection: d = ∇Φ
            d_x = -((xx - center_x) / (sigma_x**2)) * phi * 16.0
            d_y = -((yy - center_y) / (sigma_y**2)) * phi * 16.0

            # Compute pixel lookup coordinates
            map_x = xx + d_x
            map_y = yy + d_y

            # Warp gravel background through refractive ray displacement
            warped = cv2.remap(
                base_gravel,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            noise = np.random.randint(-2, 3, size=warped.shape, dtype=np.int16)
            frame = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()

    # Re-encode with ffmpeg to standard H.264 if available
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "main", "-level", "3.1",
            "-crf", "20", "-movflags", "+faststart",
            output_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        else:
            if os.path.exists(temp_path):
                shutil.move(temp_path, output_path)
    else:
        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)

    print(f"[OK] Successfully created: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


if __name__ == "__main__":
    generate_sample_video()
