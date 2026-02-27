#!/usr/bin/env python3
"""Speech-to-Text - Local (Whisper) or Cloud (ElevenLabs)"""
import sys
import os
import argparse
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf


def record_audio(samplerate=16000):
    """Record audio from microphone. Press Enter to stop."""
    print("Recording... (press Enter to stop)")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback)
    stream.start()
    input()
    stream.stop()
    stream.close()

    if not frames:
        print("No audio recorded.")
        sys.exit(1)

    return np.concatenate(frames), samplerate


def transcribe_local(audio, samplerate):
    """Transcribe using Whisper (local, GPU-accelerated)."""
    import whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate)
        tmp_path = f.name

    try:
        print("Transcribing with Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path, fp16=True)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def transcribe_cloud(audio, samplerate):
    """Transcribe using ElevenLabs STT API."""
    from elevenlabs import ElevenLabs

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not set. Run: source ~/.env")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate)
        tmp_path = f.name

    try:
        print("Transcribing with ElevenLabs...")
        client = ElevenLabs(api_key=api_key)
        with open(tmp_path, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
            )
        return result.text.strip()
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", action="store_true", help="Use Whisper (default)")
    group.add_argument("-c", "--cloud", action="store_true", help="Use ElevenLabs STT")
    args = parser.parse_args()

    audio, sr = record_audio()

    if args.cloud:
        text = transcribe_cloud(audio, sr)
    else:
        text = transcribe_local(audio, sr)

    print(f"\n{text}")


if __name__ == "__main__":
    main()
