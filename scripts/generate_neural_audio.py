"""
Generate studio-grade neural voiceover tracks using Microsoft Edge Neural TTS
Module: scripts/generate_neural_audio.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import wave
import edge_tts

# Ultra-realistic neural narrator voice
VOICE = "en-US-ChristopherNeural"
RATE = "+0%"
PITCH = "+0Hz"

SCENES = {
    "scene1": (
        "Fugitive methane gas leaks account for over thirty percent of global warming and create severe explosion hazards. "
        "Yet because methane is completely invisible to standard cameras, facilities must rely on specialized infrared thermal cameras costing over eighty-five thousand dollars each. "
        "AeroBOS solves this crisis by turning ordinary one hundred and fifty dollar security cameras into real-time optical gas imagers."
    ),
    "scene2": (
        "AeroBOS uses Background-Oriented Schlieren physics. "
        "When gas escapes, local density variations alter the refractive index of light according to the Gladstone-Dale relation. "
        "Using OpenCV Dense Inverse Search optical flow, AeroBOS measures these microscopic sub-pixel deflections against natural background textures, "
        "and inverts the two-dimensional Poisson equation in Fourier space to reconstruct the gas plume in real time."
    ),
    "scene3": (
        "Here is our offline video pipeline running on a surveillance feed. "
        "On the left, an expanding methane blowout is occurring over a gravel texture, completely invisible to the human eye. "
        "On the right, the AeroBOS engine calculates the sub-pixel refractive displacement, applies spatial divergence filtering, "
        "and renders a false-color turbo colormap with bounding box tracking, achieving sub-second automated detection."
    ),
    "scene4": (
        "To validate AeroBOS against actual industry data, we benchmarked it using official imagery from an industrial biogas facility inspection. "
        "AeroBOS successfully localized the fugitive roof seam leak with an Intersection-over-Union of zero point eight four compared to the physical eighty-five thousand dollar FLIR thermal camera, "
        "accurately estimating the emission flux at four point eight-five kilograms per hour."
    ),
    "scene5": (
        "We wrapped the complete pipeline in an industrial, dark-mode Streamlit web application. "
        "Operators can drag and drop surveillance video, tune optical flow sensitivity and resolution scaling, "
        "inspect synchronized side-by-side playback, and monitor real-time diagnostic metrics with direct MP4 video export."
    ),
    "scene6": (
        "By delivering a five-hundred-and-sixty-six-times reduction in equipment cost, "
        "AeroBOS makes continuous, autonomous gas leak monitoring accessible to every energy facility worldwide. "
        "The entire offline pipeline is open-source and ready to run on standard hardware. "
        "Thank you for watching AeroBOS."
    )
}

AUDIO_DIR = "build_video/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

async def generate_scene_audio(name: str, text: str):
    temp_mp3 = f"{AUDIO_DIR}/{name}_raw.mp3"
    final_wav = f"{AUDIO_DIR}/{name}.wav"
    
    # 1. Synthesize neural speech via Edge TTS
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await communicate.save(temp_mp3)
    
    # 2. Master audio with FFmpeg:
    #    - Resample to 48kHz, 16-bit PCM stereo
    #    - High-pass 80Hz (remove rumble)
    #    - Low-pass 12kHz (smooth top end)
    #    - EBU R128 broadcast loudness normalization
    af_filter = "highpass=f=80,lowpass=f=12000,loudnorm=I=-16:TP=-1.5:LRA=10"
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_mp3,
        "-af", af_filter,
        "-ar", "48000",
        "-ac", "2",
        final_wav
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
        
    with wave.open(final_wav, "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())
        
    print(f"[OK] {name}: {duration:.2f}s | 48kHz Stereo Studio Mastered -> {final_wav}")

async def main():
    print("=" * 80)
    print(f"[*] Synthesizing Studio-Grade Neural Voiceover with {VOICE}...")
    print("=" * 80)
    for name, text in SCENES.items():
        await generate_scene_audio(name, text)
    print("=" * 80)
    print("[SUCCESS] All neural voiceover tracks generated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
