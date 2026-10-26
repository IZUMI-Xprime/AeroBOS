import subprocess
import os

ps_code = """
Add-Type -AssemblyName System.Speech
$syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
$syn.SetOutputToWaveFile("test_audio.wav")
$syn.Speak("AeroBOS real time optical gas imaging presentation.")
$syn.Dispose()
"""

with open("test_tts.ps1", "w", encoding="utf-8") as f:
    f.write(ps_code)

res = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "test_tts.ps1"], capture_output=True, text=True)
print("Return code:", res.returncode)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

if os.path.exists("test_audio.wav"):
    print("SUCCESS: test_audio.wav generated! Size:", os.path.getsize("test_audio.wav"))
    os.remove("test_audio.wav")
if os.path.exists("test_tts.ps1"):
    os.remove("test_tts.ps1")
