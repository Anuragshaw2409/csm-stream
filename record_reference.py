"""
Record a short reference clip for CSM voice cloning.

Run this on your LOCAL machine (the one with a microphone), not the GPU server.
Requires: pip install sounddevice soundfile numpy

Usage:
    python record_reference.py reference.wav --seconds 10
"""
import argparse
import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Record a reference audio clip for CSM voice cloning.")
    parser.add_argument("output", help="Output wav file path, e.g. reference.wav")
    parser.add_argument("--seconds", type=float, default=10.0, help="Recording length in seconds (default 10)")
    parser.add_argument("--samplerate", type=int, default=24000, help="Sample rate (default 24000, matches CSM)")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="List available input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Recording {args.seconds}s at {args.samplerate}Hz. Speak clearly and naturally, starting in 3...")
    sd.sleep(1000)
    print("2...")
    sd.sleep(1000)
    print("1...")
    sd.sleep(1000)
    print("Recording now — GO.")

    audio = sd.rec(
        int(args.seconds * args.samplerate),
        samplerate=args.samplerate,
        channels=1,
        dtype="float32",
        device=args.device,
    )
    sd.wait()
    print("Done recording.")

    sf.write(args.output, audio, args.samplerate)
    print(f"Saved to {args.output}")
    print("Remember to also write down the EXACT transcript of what you said — that's your Reference Text.")


if __name__ == "__main__":
    main()
