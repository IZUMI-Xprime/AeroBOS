"""
AeroBOS: Verification Test Video Generator
Module: generate_test_video.py

Generates a 3-second physically grounded test video:
- Rich high-frequency textured background (checkerboard + industrial noise).
- Frames 1-20: Clean reference baseline (zero plume).
- Frames 21-90: Turbulent buoyant methane plume introducing genuine refractive
  ray deflections via cv2.remap.
- Saves to test_videos/plume_test_3s.mp4 encoded with H.264.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import cv2
import numpy as np


def generate_plume_test_video(
    output_path: str = "test_videos/plume_test_3s.mp4",
    width: int = 640,
    height: int = 360,
    fps: float = 30.0,
    duration_sec: float = 3.0,
) -> str:
    """Generates a 3-second optical gas imaging test video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = output_path.replace(".mp4", "_temp.mp4")

    total_frames = int(fps * duration_sec)

    # 1. Generate Textured Background (Industrial Concrete & Formwork Seams)
    tile = 24
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                bg[y:y+tile, x:x+tile] = [170, 175, 180]
            else:
                bg[y:y+tile, x:x+tile] = [95, 100, 105]

    # Add fine aggregate noise
    np.random.seed(42)
    noise = np.random.normal(0, 14, bg.shape).astype(np.float32)
    base_bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. Setup OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    print(f"[*] Generating {duration_sec}s test video ({total_frames} frames @ {fps} FPS)...")

    # Coordinate grids for remap
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    for frame_idx in range(total_frames):
        if frame_idx < 18:
            # Baseline period: static scene with minor sensor noise
            frame_noise = np.random.normal(0, 0.8, base_bg.shape).astype(np.float32)
            frame = np.clip(base_bg.astype(np.float32) + frame_noise, 0, 255).astype(np.uint8)
        else:
            # Plume active: convective rising methane leak
            t = (frame_idx - 18) / 30.0  # Time in seconds since blowout inception

            # Source coordinates at lower-middle
            src_x = width * 0.45
            src_y = height * 0.78

            # Plume rises and drifts slightly with wind (30 deg to the right)
            drift_x = 45.0 * t
            drift_y = -65.0 * t

            core_x = src_x + drift_x
            core_y = src_y + drift_y

            # Expanding dispersion radius
            sigma_x = 28.0 + 18.0 * math.sqrt(max(t, 0.01))
            sigma_y = 35.0 + 22.0 * math.sqrt(max(t, 0.01))

            # Turbulent pulsation
            turbulence = 1.0 + 0.25 * math.sin(t * 14.0)

            # Optical Path Difference Gaussian field: Φ(x, y)
            dist_sq = ((xx - core_x) ** 2) / (2.0 * sigma_x**2) + ((yy - core_y) ** 2) / (2.0 * sigma_y**2)
            phi = np.exp(-dist_sq) * 3.8 * turbulence

            # Refractive ray deflection: d_x = ∂Φ/∂x, d_y = ∂Φ/∂y
            # Physical Gladstone-Dale ray bending
            d_x = -((xx - core_x) / (sigma_x**2)) * phi * 12.0
            d_y = -((yy - core_y) / (sigma_y**2)) * phi * 12.0

            # Add micro-scale turbulent eddy vortices
            eddy_freq = 0.08
            eddy_phase = t * 6.0
            vortex_x = np.sin(xx * eddy_freq + eddy_phase) * np.cos(yy * eddy_freq) * 0.45 * np.exp(-dist_sq)
            vortex_y = np.cos(xx * eddy_freq) * np.sin(yy * eddy_freq + eddy_phase) * 0.45 * np.exp(-dist_sq)

            map_x = xx + d_x + vortex_x
            map_y = yy + d_y + vortex_y

            # Warp background texture using real optical deflection
            warped = cv2.remap(base_bg, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frame_noise = np.random.normal(0, 0.8, warped.shape).astype(np.float32)
            frame = np.clip(warped.astype(np.float32) + frame_noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()

    # Re-encode to H.264 using ffmpeg if available
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "main", "-level", "3.1",
            "-crf", "20", "-movflags", "+faststart",
            output_path
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

    print(f"[OK] Generated test video: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


if __name__ == "__main__":
    generate_plume_test_video()
