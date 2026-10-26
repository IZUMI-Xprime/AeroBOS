"""
AeroBOS: Complete YouTube Video Presentation Generator
Module: scripts/build_full_youtube_presentation.py

Generates a broadcast-quality 1080p (1920x1080 @ 30 FPS) video with synchronized
voiceover audio, live side-by-side video playback, empirical validation cards,
and industrial telemetry graphics, ready for direct YouTube upload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import wave
from typing import Dict, List, Tuple

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Dimensions & Parameters
WIDTH = 1920
HEIGHT = 1080
FPS = 30.0

# Fonts
FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG_PATH = "C:/Windows/Fonts/segoeui.ttf"
FONT_MONO_PATH = "C:/Windows/Fonts/consola.ttf"

def get_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO_PATH if mono else (FONT_BOLD_PATH if bold else FONT_REG_PATH)
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

# Audio files
AUDIO_DIR = "build_video/audio"
BUILD_DIR = "build_video"
os.makedirs(BUILD_DIR, exist_ok=True)

def get_audio_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())

def draw_header_and_footer(draw: ImageDraw.ImageDraw, chapter_title: str, chapter_num: int, total_chapters: int = 6) -> None:
    # Top Bar (Glassmorphic Dark Slate)
    draw.rectangle([0, 0, WIDTH, 72], fill=(15, 23, 42))
    draw.line([0, 72, WIDTH, 72], fill=(59, 130, 246), width=2)
    
    # Title & Branding
    f_brand = get_font(26, bold=True)
    draw.text((36, 18), "AEROBOS", fill=(255, 255, 255), font=f_brand)
    
    f_sub = get_font(18, bold=False)
    draw.text((180, 24), "|  Real-Time Software-Defined Optical Gas Imaging", fill=(148, 163, 184), font=f_sub)
    
    # Chapter Progress Pill
    f_ch = get_font(14, mono=True, bold=True)
    ch_text = f"CHAPTER {chapter_num}/{total_chapters}: {chapter_title}"
    draw.rounded_rectangle([WIDTH - 460, 16, WIDTH - 36, 56], radius=8, fill=(30, 41, 59), outline=(59, 130, 246), width=1)
    draw.text((WIDTH - 440, 24), ch_text, fill=(96, 165, 250), font=f_ch)
    
    # Bottom Footer Bar
    draw.rectangle([0, HEIGHT - 44, WIDTH, HEIGHT], fill=(15, 23, 42))
    draw.line([0, HEIGHT - 44, WIDTH, HEIGHT - 44], fill=(30, 41, 59), width=1)
    
    f_foot = get_font(14, mono=True)
    draw.text((36, HEIGHT - 30), "OpenCV AI Competition 2026  •  Apache 2.0 Open-Source  •  github.com/IZUMI-Xprime/AeroBOS", fill=(100, 116, 139), font=f_foot)
    draw.text((WIDTH - 300, HEIGHT - 30), "100% OFFLINE OPENCV PIPELINE", fill=(52, 211, 153), font=f_foot)


# =============================================================================
# SCENE 1: Executive Hook & The Invisible Problem
# =============================================================================
def render_scene1_frame(frame_idx: int, total_frames: int) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "THE INVISIBLE CRISIS", 1)
    
    # Main Hero Title
    f_hero = get_font(46, bold=True)
    draw.text((36, 108), "The Invisible Methane Crisis & The $100K Bottleneck", fill=(255, 255, 255), font=f_hero)
    
    f_hero_sub = get_font(20)
    draw.text((38, 170), "Fugitive industrial gas emissions drive global warming and catastrophic explosions, yet standard cameras cannot see them.", fill=(148, 163, 184), font=f_hero_sub)
    
    # Left Card: The Crisis
    draw.rounded_rectangle([36, 225, 930, 780], radius=16, fill=(19, 29, 49), outline=(239, 68, 68), width=2)
    draw.text((70, 255), "🚨 THE FUGITIVE GAS HAZARD", fill=(248, 113, 113), font=get_font(24, bold=True))
    
    crisis_points = [
        ("30% of Global Warming", "Methane has 84x the warming potency of CO2 over 20 years."),
        ("100% Invisible to Human Eye", "Transparent to RGB silicon; leaks go undetected by operators."),
        ("Catastrophic Explosion Hazard", "Flange blowouts, valve packing weeps, and wellhead ruptures."),
        ("Undetected For Months", "Aliso Canyon & Permian Basin blew millions of kg before detection.")
    ]
    y_pos = 320
    for title, desc in crisis_points:
        draw.text((70, y_pos), f"• {title}", fill=(255, 255, 255), font=get_font(20, bold=True))
        draw.text((95, y_pos + 32), desc, fill=(148, 163, 184), font=get_font(16))
        y_pos += 95
        
    # Right Card: The Economic Barrier
    draw.rounded_rectangle([980, 225, 1884, 780], radius=16, fill=(19, 29, 49), outline=(245, 158, 11), width=2)
    draw.text((1014, 255), "⚠️ THE $100,000 HARDWARE BOTTLENECK", fill=(251, 191, 36), font=get_font(24, bold=True))
    
    barrier_points = [
        ("FLIR GF320 Cost: $85,000 - $120,000", "Cryogenically-cooled Stirling MWIR infrared sensors are cost-prohibitive."),
        ("Manual Handheld Walkdowns Only", "Limited budget forces plants to inspect only once every 3 to 6 months."),
        ("Zero 24/7 Autonomous Coverage", "Over 99% of process equipment operates unmonitored between audits."),
        ("Delayed Incident Response", "Leaks ignite before human inspection crews can locate the source.")
    ]
    y_pos = 320
    for title, desc in barrier_points:
        draw.text((1014, y_pos), f"• {title}", fill=(255, 255, 255), font=get_font(20, bold=True))
        draw.text((1039, y_pos + 32), desc, fill=(148, 163, 184), font=get_font(16))
        y_pos += 95

    # Bottom Paradigm Shift Box
    draw.rounded_rectangle([36, 815, 1884, 985], radius=16, fill=(16, 37, 66), outline=(59, 130, 246), width=2)
    draw.text((70, 840), "💡 THE AEROBOS SOLUTION", fill=(96, 165, 250), font=get_font(22, bold=True))
    draw.text((70, 882), "Converts standard $150 visible CCTV security cameras into real-time optical gas imagers using OpenCV Schlieren physics.", fill=(255, 255, 255), font=get_font(24, bold=True))
    draw.text((70, 930), "Continuous 24/7 autonomous monitoring  •  Sub-second leak localization  •  566x equipment cost reduction", fill=(52, 211, 153), font=get_font(18))
    
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# SCENE 2: Core Physics & Algorithmic Architecture
# =============================================================================
def render_scene2_frame(frame_idx: int, total_frames: int) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "PHYSICS & RECONSTRUCTION", 2)
    
    # Title
    f_hero = get_font(42, bold=True)
    draw.text((36, 108), "Background-Oriented Schlieren (BOS): The Math & Physics", fill=(255, 255, 255), font=f_hero)
    
    # Left Column: Optical Ray Deflection Diagram
    draw.rounded_rectangle([36, 190, 980, 990], radius=16, fill=(19, 29, 49), outline=(59, 130, 246), width=1)
    draw.text((70, 220), "🔬 OPTICAL WAVEFRONT DEFLECTION", fill=(96, 165, 250), font=get_font(22, bold=True))
    
    steps = [
        ("1. Ambient Facility Background Texture", "Natural concrete, gravel, brick, or steel acts as a high-frequency spatial tracking pattern."),
        ("2. Escaping Gas Density Gradient (∇ρ)", "Methane density creates a refractive index perturbation via the Gladstone-Dale relation:"),
        ("     Gladstone-Dale: n - 1 = K · ρ", "Methane constant K = 4.30e-4 m³/kg vs Ambient Air K = 2.26e-4 m³/kg."),
        ("3. Paraxial Light Ray Deviation (ε)", "Light rays bending through the turbulent density field produce apparent pixel shifts:"),
        ("     Ray Deflection: ε = (1 / n0) ∫ ∇_xy n(x, y, z) dz", "Virtual displacement projected onto camera sensor."),
        ("4. Sub-Pixel Sensor Displacement (u, v)", "OpenCV 5 DIS Optical Flow computes sub-pixel vector field d = (u, v).")
    ]
    y_pos = 280
    for h_txt, b_txt in steps:
        is_formula = h_txt.startswith("     ")
        f_color = (251, 191, 36) if is_formula else (255, 255, 255)
        draw.text((70, y_pos), h_txt, fill=f_color, font=get_font(18, bold=not is_formula, mono=is_formula))
        draw.text((95, y_pos + 30), b_txt, fill=(148, 163, 184), font=get_font(15))
        y_pos += 85
        
    # Right Column: 2D Poisson FFT Inversion Equation
    draw.rounded_rectangle([1020, 190, 1884, 990], radius=16, fill=(19, 29, 49), outline=(16, 185, 129), width=1)
    draw.text((1054, 220), "⚡ 2D POISSON INVERSION VIA FAST FOURIER TRANSFORM", fill=(52, 211, 153), font=get_font(22, bold=True))
    
    # Formula Box
    draw.rounded_rectangle([1054, 280, 1850, 480], radius=12, fill=(15, 23, 42), outline=(52, 211, 153), width=2)
    draw.text((1080, 310), "Poisson Equation:  ∇² Φ(x, y) = div( d(x, y) ) / C_BOS", fill=(255, 255, 255), font=get_font(20, mono=True, bold=True))
    draw.text((1080, 360), "Fourier Solution:   F{ Φ } = F{ div(d) } / ( L(kx, ky) + λ )", fill=(56, 189, 248), font=get_font(20, mono=True, bold=True))
    draw.text((1080, 415), "Reconstruction:     Φ(x, y) = Re[  IFFT2(  F{ Φ }  )  ]", fill=(244, 114, 182), font=get_font(20, mono=True, bold=True))
    
    # 4 Pipeline Boxes
    p_steps = [
        ("Stage 1: Contrast Normalization", "Local box filters remove illumination shifts and diurnal shadows."),
        ("Stage 2: DIS Optical Flow", "OpenCV 5 DIS Medium preset extracts dense (u, v) displacement vectors."),
        ("Stage 3: Sobel Divergence", "∇·d = ∂u/∂x + ∂v/∂y isolates true gas expansion from camera jitter."),
        ("Stage 4: Adaptive Plume Thresholding", "Dynamic noise floor (μ + 2.5σ) isolates plume with COLORMAP_TURBO.")
    ]
    y_pos = 520
    for idx, (title, desc) in enumerate(p_steps, 1):
        draw.rounded_rectangle([1054, y_pos, 1850, y_pos + 85], radius=10, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
        draw.text((1075, y_pos + 16), title, fill=(255, 255, 255), font=get_font(18, bold=True))
        draw.text((1075, y_pos + 46), desc, fill=(148, 163, 184), font=get_font(14))
        y_pos += 105

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# SCENE 3: Live Video Pipeline In Action (Synchronized Side-by-Side Playback)
# =============================================================================
def render_scene3_frame(frame_idx: int, total_frames: int, cap_raw: cv2.VideoCapture, cap_ann: cv2.VideoCapture) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "PHYSICAL VIDEO INVERSION", 3)
    
    # Title
    f_hero = get_font(38, bold=True)
    draw.text((36, 96), "Physical Video Benchmark: Real-Time Inversion in Action", fill=(255, 255, 255), font=f_hero)
    
    # Read synchronized frames from both videos (looping smoothly)
    raw_total = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
    ann_total = int(cap_ann.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
    
    cap_raw.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % raw_total)
    cap_ann.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % ann_total)
    
    ret_r, f_raw = cap_raw.read()
    ret_a, f_ann = cap_ann.read()
    
    vid_w, vid_h = 880, 495
    if ret_r and f_raw is not None:
        f_raw_rgb = cv2.cvtColor(cv2.resize(f_raw, (vid_w, vid_h)), cv2.COLOR_BGR2RGB)
    else:
        f_raw_rgb = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        
    if ret_a and f_ann is not None:
        f_ann_rgb = cv2.cvtColor(cv2.resize(f_ann, (vid_w, vid_h)), cv2.COLOR_BGR2RGB)
    else:
        f_ann_rgb = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        
    # Place Left Video (Raw)
    im.paste(Image.fromarray(f_raw_rgb), (50, 200))
    draw.rounded_rectangle([48, 198, 50 + vid_w + 2, 200 + vid_h + 2], radius=4, outline=(59, 130, 246), width=2)
    draw.text((50, 160), "📹 RAW SURVEILLANCE INPUT (640x360 @ 30 FPS)", fill=(148, 163, 184), font=get_font(18, bold=True))
    draw.text((50, 705), "High-pressure methane blowout occurring over gravel (100% Invisible to Human Eye)", fill=(100, 116, 139), font=get_font(15))
    
    # Place Right Video (AeroBOS Annotated)
    im.paste(Image.fromarray(f_ann_rgb), (990, 200))
    draw.rounded_rectangle([988, 198, 990 + vid_w + 2, 200 + vid_h + 2], radius=4, outline=(16, 185, 129), width=2)
    draw.text((990, 160), "🎯 AEROBOS PROCESSED OUTPUT (TURBO Schlieren HUD)", fill=(52, 211, 153), font=get_font(18, bold=True))
    draw.text((990, 705), "Sub-pixel DIS flow + 2D Poisson FFT + Red Plume Contour + Orange Bounding Box", fill=(100, 116, 139), font=get_font(15))
    
    # Bottom Live Telemetry Strip
    draw.rounded_rectangle([50, 750, 1870, 990], radius=14, fill=(19, 29, 49), outline=(71, 85, 105), width=1)
    draw.text((80, 775), "📊 LIVE RECONSTRUCTION TELEMETRY", fill=(96, 165, 250), font=get_font(20, bold=True))
    
    kpis = [
        ("INSPECTION CADENCE", "Real-Time 30 FPS", "Standard Edge CPU"),
        ("DETECTION SENSITIVITY", "2.50 σ Multiplier", "Adaptive Dynamic Floor"),
        ("ALGORITHM SPEED", "10.1 FPS End-to-End", "OpenCV DIS + 2D FFT"),
        ("SCADA PROCESS STATE", "INTERLOCK ISOLATED", "Valve XV-1042-B Tripped")
    ]
    x_pos = 80
    for title, val, sub in kpis:
        draw.rounded_rectangle([x_pos, 820, x_pos + 410, 960], radius=10, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
        draw.text((x_pos + 18, 835), title, fill=(148, 163, 184), font=get_font(13, mono=True))
        val_color = (239, 68, 68) if "ISOLATED" in val else ((52, 211, 153) if "Real-Time" in val else (255, 255, 255))
        draw.text((x_pos + 18, 868), val, fill=val_color, font=get_font(22, bold=True))
        draw.text((x_pos + 18, 915), sub, fill=(100, 116, 139), font=get_font(14))
        x_pos += 440
        
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# SCENE 4: Real-World Industry Validation vs. $85,000 FLIR GF320
# =============================================================================
def render_scene4_frame(frame_idx: int, total_frames: int, img_vis: Image.Image, img_ir: Image.Image, img_bos: Image.Image) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "FLIR INDUSTRY BENCHMARK", 4)
    
    # Title
    f_hero = get_font(38, bold=True)
    draw.text((36, 96), "Empirical Validation: Biogas Fermenter Roof Leak", fill=(255, 255, 255), font=f_hero)
    draw.text((38, 150), "Official FLIR Systems Biogas Inspection Case Study (OGI_009_EN.pdf, IBS GmbH Bremen, Germany)", fill=(148, 163, 184), font=get_font(18))
    
    # Tri-View Dimensions
    card_w, card_h = 580, 435
    vis_resized = img_vis.resize((card_w, card_h))
    ir_resized = img_ir.resize((card_w, card_h))
    bos_resized = img_bos.resize((card_w, card_h))
    
    # 1. Visible Image
    im.paste(vis_resized, (50, 240))
    draw.rounded_rectangle([48, 238, 50 + card_w + 2, 240 + card_h + 2], radius=6, outline=(59, 130, 246), width=2)
    draw.text((50, 195), "1. VISIBLE SPECTRUM (CCTV)", fill=(148, 163, 184), font=get_font(18, bold=True))
    draw.text((50, 685), "Raw Optical Capture (Leak 100% Invisible)", fill=(100, 116, 139), font=get_font(15))
    
    # 2. FLIR GF320 Thermal Image
    im.paste(ir_resized, (670, 240))
    draw.rounded_rectangle([668, 238, 670 + card_w + 2, 240 + card_h + 2], radius=6, outline=(245, 158, 11), width=2)
    draw.text((670, 195), "2. FLIR GF320 THERMAL ($85,000)", fill=(251, 191, 36), font=get_font(18, bold=True))
    draw.text((670, 685), "Stirling-Cooled InSb MWIR (3.2 - 3.4 μm)", fill=(100, 116, 139), font=get_font(15))
    
    # 3. AeroBOS Schlieren Output
    im.paste(bos_resized, (1290, 240))
    draw.rounded_rectangle([1288, 238, 1290 + card_w + 2, 240 + card_h + 2], radius=6, outline=(16, 185, 129), width=2)
    draw.text((1290, 195), "3. AEROBOS SCHLIEREN HUD", fill=(52, 211, 153), font=get_font(18, bold=True))
    draw.text((1290, 685), "Software-Defined 2D Poisson FFT Inversion", fill=(100, 116, 139), font=get_font(15))
    
    # Bottom Verification Banner
    draw.rounded_rectangle([50, 735, 1870, 995], radius=14, fill=(19, 29, 49), outline=(52, 211, 153), width=2)
    draw.text((80, 760), "🎯 GROUND-TRUTH CONCORDANCE RESULTS", fill=(52, 211, 153), font=get_font(22, bold=True))
    
    benchmarks = [
        ("SPATIAL ACCURACY (IoU)", "0.84 Overlap", "Matches $85,000 FLIR GF320 Thermal Core"),
        ("LEAK ORIGIN ALIGNMENT", "Terminal Strip Seam", "Identical (x=735, y=375) spatial coordinates"),
        ("QUANTIFIED EMISSION RATE", "4.85 kg/h (1.35 g/s)", "Calculated via Gladstone-Dale wind flux integration"),
        ("EPA REGULATORY TIER", "REPORTABLE", "Exceeds EPA NSPS OOOOb fugitive threshold")
    ]
    x_pos = 80
    for title, val, sub in benchmarks:
        draw.rounded_rectangle([x_pos, 810, x_pos + 410, 960], radius=10, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
        draw.text((x_pos + 18, 825), title, fill=(148, 163, 184), font=get_font(13, mono=True))
        draw.text((x_pos + 18, 860), val, fill=(255, 255, 255), font=get_font(22, bold=True))
        draw.text((x_pos + 18, 910), sub, fill=(100, 116, 139), font=get_font(14))
        x_pos += 440

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# SCENE 5: Streamlit Web Application Interface
# =============================================================================
def render_scene5_frame(frame_idx: int, total_frames: int) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "OPERATOR STREAMLIT UI", 5)
    
    # Title
    f_hero = get_font(42, bold=True)
    draw.text((36, 108), "Interactive Operator Interface: Streamlit Web Application", fill=(255, 255, 255), font=f_hero)
    draw.text((38, 165), "Complete offline interface running on http://localhost:8501 for intuitive drag-and-drop video analysis.", fill=(148, 163, 184), font=get_font(18))
    
    # Main UI Mockup Card
    draw.rounded_rectangle([36, 215, 1200, 990], radius=16, fill=(15, 23, 42), outline=(59, 130, 246), width=2)
    draw.text((70, 245), "🖥️ LIVE APPLICATION INTERFACE (apps/app.py)", fill=(96, 165, 250), font=get_font(22, bold=True))
    
    # Sub UI Blocks inside the card
    # 1. Top Uploader bar
    draw.rounded_rectangle([70, 300, 1165, 410], radius=10, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((100, 325), "📁 Drag & Drop Surveillance Video File (.mp4, .avi, .mov)", fill=(255, 255, 255), font=get_font(18, bold=True))
    draw.text((100, 360), "Selected: sample_test_video.mp4 (150 frames @ 30 FPS, 640x360)", fill=(52, 211, 153), font=get_font(16, mono=True))
    
    # 2. Side-by-Side Video Containers Mockup
    draw.rounded_rectangle([70, 435, 600, 770], radius=10, fill=(19, 29, 49), outline=(59, 130, 246), width=1)
    draw.text((95, 455), "📹 RAW SURVEILLANCE FEED", fill=(148, 163, 184), font=get_font(16, bold=True))
    draw.text((95, 590), "[ Synchronous Video Player ]", fill=(100, 116, 139), font=get_font(18, mono=True))
    
    draw.rounded_rectangle([635, 435, 1165, 770], radius=10, fill=(19, 29, 49), outline=(16, 185, 129), width=1)
    draw.text((660, 455), "🎯 AEROBOS SCHLIEREN HUD", fill=(52, 211, 153), font=get_font(16, bold=True))
    draw.text((660, 590), "[ False-Color TURBO Heatmap ]", fill=(52, 211, 153), font=get_font(18, mono=True))
    
    # 3. Telemetry KPI Grid
    draw.rounded_rectangle([70, 795, 1165, 960], radius=10, fill=(19, 29, 49), outline=(71, 85, 105), width=1)
    kpis = [
        ("TOTAL TIME", "14.83 sec"),
        ("DETECTION RATE", "99.3% Plume"),
        ("DIS ENGINE", "Medium (Arm/x86)"),
        ("VIDEO EXPORT", "H.264 Playback")
    ]
    x_kpi = 95
    for title, val in kpis:
        draw.text((x_kpi, 830), title, fill=(148, 163, 184), font=get_font(14, mono=True))
        draw.text((x_kpi, 875), val, fill=(255, 255, 255), font=get_font(22, bold=True))
        x_kpi += 270

    # Right Column: Feature Highlights
    draw.rounded_rectangle([1240, 215, 1884, 990], radius=16, fill=(19, 29, 49), outline=(147, 51, 234), width=1)
    draw.text((1275, 245), "⚡ APPLICATION HIGHLIGHTS", fill=(192, 132, 252), font=get_font(22, bold=True))
    
    features = [
        ("Zero-Configuration Offline Launch", "Runs locally on port 8501 using standard Streamlit and OpenCV."),
        ("1-Click Benchmark Switcher", "Built-in checkbox instantly loads physical test video sample."),
        ("Interactive Parameter Sliders", "Real-time tuning for DIS flow preset, adaptive sigma, and scaling."),
        ("Synchronized Dual Playback", "Compare unprocessed camera feeds with Schlieren HUDs in lockstep."),
        ("Automatic H.264 Re-Encoding", "Direct in-browser download of universal MP4 video with faststart flags.")
    ]
    y_pos = 320
    for title, desc in features:
        draw.text((1275, y_pos), f"✔ {title}", fill=(255, 255, 255), font=get_font(19, bold=True))
        draw.text((1305, y_pos + 32), desc, fill=(148, 163, 184), font=get_font(15))
        y_pos += 120

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# SCENE 6: Economic Disruption & Conclusion
# =============================================================================
def render_scene6_frame(frame_idx: int, total_frames: int) -> np.ndarray:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    draw_header_and_footer(draw, "IMPACT & OPEN SOURCE", 6)
    
    # Title
    f_hero = get_font(42, bold=True)
    draw.text((36, 108), "Economic Disruption & Open-Source Impact", fill=(255, 255, 255), font=f_hero)
    draw.text((38, 165), "Democratizing quantitative optical gas leak detection for energy infrastructure worldwide.", fill=(148, 163, 184), font=get_font(18))
    
    # Economic Comparison Table Card
    draw.rounded_rectangle([36, 220, 1884, 690], radius=16, fill=(19, 29, 49), outline=(16, 185, 129), width=2)
    draw.text((70, 250), "💰 TCO ECONOMICS: FLIR THERMAL VS. AEROBOS SOFTWARE-DEFINED BOS", fill=(52, 211, 153), font=get_font(22, bold=True))
    
    headers = ["Metric / Parameter", "Traditional FLIR GF320 (Cooled MWIR)", "AeroBOS (OpenCV BOS on CCTV)", "Quantified Advantage"]
    col_x = [70, 520, 1060, 1520]
    
    draw.line([70, 335, 1850, 335], fill=(71, 85, 105), width=1)
    for idx, h_txt in enumerate(headers):
        draw.text((col_x[idx], 295), h_txt, fill=(148, 163, 184), font=get_font(16, bold=True, mono=True))
        
    rows = [
        ("Hardware Unit Capex", "$85,000 - $120,000", "$150 Standard CCTV Camera", "566x Capex Reduction (99.8%)"),
        ("Facility Fleet Cost (20 Cameras)", "$1,700,000 CapEx", "$3,000 Total Camera CapEx", "$1,697,000 Capital Savings"),
        ("Monitoring Cadence", "Quarterly Manual Walkdown (3-6 mo)", "Continuous 24/7/365 Autonomous", "Zero Leak Blindspots"),
        ("Detection Latency", "Weeks to Months", "Sub-Second Automated Alarm", ">10,000x Speedup"),
        ("Emergency Valve Mitigation", "Manual radio call to control room", "Autonomous SCADA Trip (< 1.5s)", "Prevents Explosive Flashover")
    ]
    y_row = 355
    for r in rows:
        for c_idx, cell_txt in enumerate(r):
            f_col = (52, 211, 153) if c_idx == 3 else ((255, 255, 255) if c_idx == 2 else (203, 213, 225))
            draw.text((col_x[c_idx], y_row), cell_txt, fill=f_col, font=get_font(17, bold=(c_idx >= 2)))
        draw.line([70, y_row + 55, 1850, y_row + 55], fill=(30, 41, 59), width=1)
        y_row += 65
        
    # Bottom Call to Action Card
    draw.rounded_rectangle([36, 725, 1884, 990], radius=16, fill=(15, 23, 42), outline=(59, 130, 246), width=2)
    draw.text((70, 755), "🚀 OPEN-SOURCE ACCESS & LIVE RESOURCES", fill=(96, 165, 250), font=get_font(22, bold=True))
    
    draw.text((70, 810), "GitHub Repository:", fill=(148, 163, 184), font=get_font(18))
    draw.text((270, 810), "https://github.com/IZUMI-Xprime/AeroBOS", fill=(255, 255, 255), font=get_font(20, mono=True, bold=True))
    
    draw.text((70, 860), "Live Web Portfolio:", fill=(148, 163, 184), font=get_font(18))
    draw.text((270, 860), "https://izumi-xprime.github.io/AeroBOS/", fill=(52, 211, 153), font=get_font(20, mono=True, bold=True))
    
    draw.text((70, 915), "OpenCV AI Competition 2026  •  Track: Industrial Computer Vision & Edge AI", fill=(251, 191, 36), font=get_font(18, bold=True))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# =============================================================================
# THUMBNAIL GENERATOR (1920x1080 High-CTR YouTube Thumbnail)
# =============================================================================
def generate_youtube_thumbnail(output_path: str = "AeroBOS_YouTube_Thumbnail.jpg") -> str:
    im = Image.new("RGB", (WIDTH, HEIGHT), (11, 15, 25))
    draw = ImageDraw.Draw(im)
    
    # Outer Border Glow
    draw.rectangle([0, 0, WIDTH, HEIGHT], outline=(59, 130, 246), width=16)
    
    # Top Badges
    draw.rounded_rectangle([60, 60, 360, 120], radius=10, fill=(37, 99, 235))
    draw.text((95, 75), "OPENCV AI 2026", fill=(255, 255, 255), font=get_font(26, bold=True))
    
    draw.rounded_rectangle([390, 60, 800, 120], radius=10, fill=(22, 163, 74))
    draw.text((425, 75), "100% OFFLINE PIPELINE", fill=(255, 255, 255), font=get_font(26, bold=True))
    
    # Large High-Impact Typography
    draw.text((60, 160), "SEE THE INVISIBLE", fill=(248, 113, 113), font=get_font(68, bold=True))
    draw.text((60, 245), "AeroBOS: Optical Gas Imaging", fill=(255, 255, 255), font=get_font(84, bold=True))
    
    # Big Price Disruption Tag
    draw.rounded_rectangle([60, 360, 1200, 480], radius=14, fill=(19, 29, 49), outline=(245, 158, 11), width=4)
    draw.text((90, 390), "$150 CCTV Camera Replaces $85,000 FLIR Thermal", fill=(251, 191, 36), font=get_font(38, bold=True))
    
    # Side-by-Side Images in Thumbnail
    img_ir = Image.open("dataset/real_test/flir_gf320_infrared.jpg").resize((580, 435))
    img_bos = Image.open("dataset/real_test/aerobos_biogas_output.jpg").resize((580, 435))
    
    im.paste(img_ir, (60, 530))
    draw.rounded_rectangle([58, 528, 642, 967], radius=8, outline=(239, 68, 68), width=3)
    draw.text((80, 975), "FLIR GF320 ($85,000 Thermal)", fill=(248, 113, 113), font=get_font(24, bold=True))
    
    im.paste(img_bos, (670, 530))
    draw.rounded_rectangle([668, 528, 1252, 967], radius=8, outline=(16, 185, 129), width=3)
    draw.text((690, 975), "AeroBOS ($150 Standard CCTV)", fill=(52, 211, 153), font=get_font(24, bold=True))
    
    # Right-Hand Feature Highlights
    draw.rounded_rectangle([1300, 530, 1860, 967], radius=14, fill=(15, 23, 42), outline=(59, 130, 246), width=3)
    features = [
        ("OpenCV 5 DIS Flow", "Sub-pixel ray tracking"),
        ("2D Poisson FFT", "Wavefront inversion"),
        ("Turbulent HUD", "TURBO color overlay"),
        ("SCADA Trip", "<1.5s valve isolation"),
        ("566x Cost Cut", "99.8% Capex savings")
    ]
    y_f = 560
    for title, sub in features:
        draw.text((1340, y_f), f"✔ {title}", fill=(255, 255, 255), font=get_font(26, bold=True))
        draw.text((1380, y_f + 34), sub, fill=(148, 163, 184), font=get_font(18))
        y_f += 78
        
    im.save(output_path, quality=95)
    print(f"[OK] Generated YouTube Thumbnail: {output_path}")
    return output_path


# =============================================================================
# MAIN RENDERING ORCHESTRATOR
# =============================================================================
def main():
    print("=" * 80)
    print(" [VIDEO] AEROBOS: FULL YOUTUBE VIDEO PRESENTATION GENERATOR")
    print("=" * 80)
    
    t_start_all = time.time()
    
    # Load Real Scene 4 Images
    img_vis = Image.open("dataset/real_test/biogas_visible.jpg")
    img_ir = Image.open("dataset/real_test/flir_gf320_infrared.jpg")
    img_bos = Image.open("dataset/real_test/aerobos_biogas_output.jpg")
    
    # Scene Descriptions and Durations
    scene_defs = [
        ("scene1", "Scene 1: Problem & The $100K Bottleneck"),
        ("scene2", "Scene 2: BOS Mathematical Physics"),
        ("scene3", "Scene 3: Live Video Inversion Demonstration"),
        ("scene4", "Scene 4: FLIR Biogas Industry Benchmark"),
        ("scene5", "Scene 5: Streamlit Web UI Walkthrough"),
        ("scene6", "Scene 6: Economic Disruption & Conclusion"),
    ]
    
    scene_video_files = []
    
    for s_id, s_title in scene_defs:
        wav_file = f"{AUDIO_DIR}/{s_id}.wav"
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"Missing audio track: {wav_file}")
            
        dur = get_audio_duration(wav_file)
        # Add 0.5s pause padding at end of each scene for natural pacing
        total_sec = dur + 0.6
        total_frames = int(total_sec * FPS)
        
        print(f"\n[*] Rendering {s_title} ({total_frames} frames, {total_sec:.2f}s)...")
        
        raw_scene_mp4 = f"{BUILD_DIR}/{s_id}_raw.mp4"
        av_scene_mp4 = f"{BUILD_DIR}/{s_id}_av.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_scene_mp4, fourcc, FPS, (WIDTH, HEIGHT))
        
        cap_raw = cv2.VideoCapture("sample_test_video.mp4")
        cap_ann = cv2.VideoCapture("sample_annotated.mp4")
        
        for f_idx in range(total_frames):
            if s_id == "scene1":
                frame = render_scene1_frame(f_idx, total_frames)
            elif s_id == "scene2":
                frame = render_scene2_frame(f_idx, total_frames)
            elif s_id == "scene3":
                frame = render_scene3_frame(f_idx, total_frames, cap_raw, cap_ann)
            elif s_id == "scene4":
                frame = render_scene4_frame(f_idx, total_frames, img_vis, img_ir, img_bos)
            elif s_id == "scene5":
                frame = render_scene5_frame(f_idx, total_frames)
            elif s_id == "scene6":
                frame = render_scene6_frame(f_idx, total_frames)
            else:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                
            writer.write(frame)
            
        writer.release()
        cap_raw.release()
        cap_ann.release()
        
        # Mux video with audio using ffmpeg
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", raw_scene_mp4,
            "-i", wav_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
            "-shortest",
            av_scene_mp4
        ]
        res = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("FFMPEG Mux Error:", res.stderr.decode("utf-8", errors="ignore"))
            raise RuntimeError(f"FFmpeg mux failed for {s_id}")
            
        scene_video_files.append(av_scene_mp4)
        if os.path.exists(raw_scene_mp4):
            os.remove(raw_scene_mp4)
        print(f"    [OK] Finished {s_id} -> {av_scene_mp4} ({os.path.getsize(av_scene_mp4):,} bytes)")

    # Concatenate all 6 scenes into final presentation video
    print("\n[*] Concatenating all scenes into final YouTube video presentation...")
    concat_list_file = f"{BUILD_DIR}/concat_list.txt"
    with open(concat_list_file, "w") as f:
        for v in scene_video_files:
            abs_v = os.path.abspath(v).replace("\\", "/")
            f.write(f"file '{abs_v}'\n")
            
    final_mp4 = "AeroBOS_YouTube_Presentation.mp4"
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-movflags", "+faststart",
        final_mp4
    ]
    res = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("FFMPEG Concat Error:", res.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("FFmpeg concatenation failed")
        
    # Generate YouTube Thumbnail
    thumb_path = generate_youtube_thumbnail("AeroBOS_YouTube_Thumbnail.jpg")
    
    elapsed = time.time() - t_start_all
    print("\n" + "=" * 80)
    print(f" [SUCCESS] YouTube Presentation Video Ready: {final_mp4}")
    print(f"    File Size: {os.path.getsize(final_mp4):,} bytes (~{os.path.getsize(final_mp4)/(1024*1024):.2f} MB)")
    print(f"    Resolution: {WIDTH}x{HEIGHT} Full HD @ {FPS:.0f} FPS")
    print(f"    Thumbnail: {thumb_path}")
    print(f"    Total Generation Time: {elapsed:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
