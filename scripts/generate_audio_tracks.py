"""
Generate all voiceover tracks safely escaping quotes.
"""
import subprocess
import os
import wave

SCENES = {
    "scene1": "Fugitive methane gas leaks account for over thirty percent of global warming and create severe explosion hazards. But because methane is completely invisible to standard cameras, facilities must rely on specialized thermal cameras costing over eighty-five thousand dollars each. AeroBOS solves this crisis by turning ordinary one hundred and fifty dollar security cameras into real-time optical gas imagers.",
    "scene2": "AeroBOS uses Background-Oriented Schlieren physics. When gas escapes, local density variations alter the refractive index of light according to the Gladstone-Dale relation. Using OpenCV Dense Inverse Search optical flow, AeroBOS measures these microscopic sub-pixel deflections against natural background textures, and inverts the two-dimensional Poisson equation in Fourier space to reconstruct the gas plume in real time.",
    "scene3": "Here is our offline video pipeline running on a surveillance feed. On the left, an expanding methane blowout is occurring over a gravel texture, completely invisible to the human eye. On the right, the AeroBOS engine calculates the sub-pixel refractive displacement, applies spatial divergence filtering, and renders a false-color turbo colormap with bounding box tracking, achieving sub-second automated detection.",
    "scene4": "To validate AeroBOS against actual industry data, we benchmarked it using official imagery from an industrial biogas facility inspection. AeroBOS successfully localized the fugitive roof seam leak with an Intersection-over-Union of point eight-four compared to the physical eighty-five thousand dollar FLIR thermal camera, accurately estimating the emission flux at four point eight-five kilograms per hour.",
    "scene5": "We wrapped the complete pipeline in an industrial, dark-mode Streamlit web application. Operators can drag and drop surveillance video, tune optical flow sensitivity and resolution scaling, inspect synchronized side-by-side playback, and monitor real-time diagnostic metrics with direct MP4 video export.",
    "scene6": "By delivering a five-hundred-and-sixty-six-times reduction in equipment cost, AeroBOS makes continuous, autonomous gas leak monitoring accessible to every facility worldwide. The entire offline pipeline is open-source and ready to run on standard hardware. Thank you for watching AeroBOS."
}

os.makedirs("build_video/audio", exist_ok=True)

for name, text in SCENES.items():
    wav_path = os.path.abspath(f"build_video/audio/{name}.wav").replace("\\", "/")
    # Escape double quotes and dollar signs for PowerShell
    safe_text = text.replace('"', '`"').replace('$', '`$')
    ps_code = f"""
Add-Type -AssemblyName System.Speech
$syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
$syn.Rate = 0
$syn.SetOutputToWaveFile("{wav_path}")
$syn.Speak("{safe_text}")
$syn.Dispose()
"""
    ps_file = f"build_video/audio/{name}.ps1"
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write(ps_code)
    
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_file], capture_output=True)
    if os.path.exists(ps_file):
        os.remove(ps_file)
    
    if os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
            print(f"{name}: {dur:.2f} seconds ({os.path.getsize(wav_path)} bytes)")
    else:
        print(f"FAILED to generate {name}")
