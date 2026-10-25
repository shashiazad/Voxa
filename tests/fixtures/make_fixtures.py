"""Regenerate WAV fixtures (macOS `say` + afconvert). 48 kHz mono int16."""
import subprocess, sys, tempfile, wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
LINES = {
    "status": "I would like to check the status of my order",
    "interrupt": "Wait, stop. I have a different question about my invoice",
    "backchannel": "Yeah, okay",
    "q_returns": "What is your return policy for headphones?",
    "q_order": "Where is my order A C 4 8 2 1 3?",
    "q_ticket": "My headphones are broken. Please open a ticket for order A C 4 8 2 1 3.",
    "q_yes": "Yes, please do.",
    "q_no": "No thank you, I would rather not.",
    "q_human": "I want to talk to a human.",
    "pause_number": "My order number is [[slnc 1000]] four five six seven eight",
}


def tts(name: str, text: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        aiff = f"{d}/a.aiff"
        rate = ["-r", "130"] if name in ("q_order", "q_ticket") else []   # spelled IDs: slower, as callers do
        subprocess.run(["say", "-v", "Samantha", *rate, "-o", aiff, text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@48000", "-c", "1", aiff, str(OUT / f"{name}_48k.wav")], check=True)


def write(name: str, x: np.ndarray) -> None:
    with wave.open(str(OUT / f"{name}_48k.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def synth_noises() -> None:
    rng = np.random.default_rng(7)
    n = int(0.45 * 48000)
    env = np.exp(-np.arange(n) / (0.12 * 48000))
    cough = rng.standard_normal(n) * env * 9000
    cough *= (np.sin(2 * np.pi * 14 * np.arange(n) / 48000) > -0.2)       # chesty pulsing
    write("cough", cough)
    m = int(0.35 * 48000)
    slam = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.04 * 48000)) * 20000
    slam += np.sin(2 * np.pi * 90 * np.arange(m) / 48000) * np.exp(-np.arange(m) / (0.08 * 48000)) * 15000
    write("door_slam", slam)


if __name__ == "__main__" and "--noise" not in sys.argv:
    for k, v in LINES.items():
        tts(k, v)
    synth_noises()


# ---------------------------------------------------------------- background noises (60 s, mixed at runtime at a target SNR)
BABBLE = ["I was thinking we could meet at the corner around six and then head over to the theatre together",
          "Honestly the traffic this morning was unbelievable, I sat there for forty minutes doing nothing at all",
          "Can I get a large oat latte and one of those almond croissants please, and could you put it in a cup",
          "She said the quarterly numbers look better than expected but nobody wants to say it out loud yet",
          "We booked the flights last week and the hotel is right by the beach so the kids are really excited"]
NEWS = ["Markets closed higher on Tuesday as investors weighed new data on inflation and the outlook for interest rates across the region",
        "Officials said the storm is expected to weaken overnight, though heavy rain and strong winds could continue through the weekend"]


def _say_array(voice: str, text: str, rate: int = 175) -> np.ndarray:
    import wave as w
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", f"{d}/a.aiff", text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@48000", "-c", "1", f"{d}/a.aiff", f"{d}/a.wav"], check=True)
        with w.open(f"{d}/a.wav") as f:
            return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)


def _loop(x: np.ndarray, n: int, shift: int = 0) -> np.ndarray:
    return np.resize(np.roll(x, shift), n)


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    k = np.exp(-2 * np.pi * hz / 48000)
    y = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x):
        acc = k * acc + (1 - k) * v; y[i] = acc
    return y


def synth_background() -> None:
    rng = np.random.default_rng(11)
    n = 60 * 48000
    voices = ["Daniel", "Karen", "Moira", "Tessa", "Fred"]
    babble = sum(_loop(_say_array(v, t), n, rng.integers(0, 48000 * 20)) * 0.6 for v, t in zip(voices, BABBLE))
    babble += rng.standard_normal(n) * 300                                    # room tone
    for _ in range(40):                                                       # cutlery clinks
        i = rng.integers(0, n - 4000); t = np.arange(3000)
        babble[i:i + 3000] += np.sin(2 * np.pi * rng.uniform(2500, 5000) * t / 48000) * np.exp(-t / 400) * 2500
    write("noise_cafe", babble)
    tv = _loop(_say_array("Ralph", NEWS[0], 190), n) + 0.8 * _loop(_say_array("Kathy", NEWS[1], 185), n, 90000)
    tv += 0.15 * np.sin(2 * np.pi * 220 * np.arange(n) / 48000) * (np.sin(2 * np.pi * 0.3 * np.arange(n) / 48000) > 0)   # music bed
    write("noise_tv", tv)
    brown = np.cumsum(rng.standard_normal(n)); brown -= np.convolve(brown, np.ones(4800) / 4800, mode="same")
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * np.arange(n) / (48000 * 7.0))
    write("noise_traffic", _lowpass(brown, 900) * swell * 6 + rng.standard_normal(n) * 120)
    kb = np.zeros(n); t = 0
    while t < n - 48000:
        t += int(48000 * rng.uniform(0.07, 0.22))
        L = rng.integers(120, 260); burst = rng.standard_normal(L) * np.exp(-np.arange(L) / 45)
        kb[t:t + L] += burst * rng.uniform(4000, 9000)
    write("noise_keyboard", kb)


if __name__ == "__main__" and "--noise" in sys.argv:
    synth_background()
