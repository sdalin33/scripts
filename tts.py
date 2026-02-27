#!/usr/bin/env python3
"""ElevenLabs TTS - Text to Speech"""
import sys
import os
import subprocess
import tempfile
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def speak(text, voice="21m00Tcm4TlvDq8ikWAM"):
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio:
            f.write(chunk)
        f.flush()
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", f.name])
        os.unlink(f.name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to speak: ")
    speak(text)
