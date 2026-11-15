#!/usr/bin/env python3
"""Writes a deterministic synthetic 44.1 kHz test mix (speech + chords +
percussion) for the parity checks: no copyrighted music in the repo.

usage: make_test_audio.py <speech.wav> <out.wav> [--seconds 20] [--mono] [--stems PREFIX]

--stems PREFIX also writes PREFIX_vocals.wav and PREFIX_instrumental.wav
(the true parts, same scaling: they sum to the mix).
"""
import argparse

import numpy as np
import soundfile as sf

SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speech")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--stems")
    a = ap.parse_args()
    n = int(a.seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(1234)

    sp, sr = sf.read(a.speech, dtype="float32", always_2d=True)
    sp = sp.mean(1)
    # linear-interpolation resampling is enough for a test signal
    pos = np.arange(int(len(sp) * SR / sr)) * sr / SR
    sp = np.interp(pos, np.arange(len(sp)), sp)
    voice = np.resize(sp, n) * 0.6

    chords = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63)]
    music = np.zeros(n)
    for k in range(n // SR + 1):
        f = chords[k % 3]
        s, e = k * SR, min((k + 1) * SR, n)
        tt = t[s:e]
        for fr in f:
            music[s:e] += 0.08 * (2 * ((tt * fr) % 1.0) - 1)  # saw
    beat = np.zeros(n)
    for s in range(0, n, SR // 2):
        L = min(4000, n - s)
        beat[s : s + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / 600) * 0.3
        beat[s : s + L] += np.sin(2 * np.pi * 60 * np.arange(L) / SR) * np.exp(-np.arange(L) / 3000) * 0.5

    v = np.stack([voice, voice * 0.9], 1)
    acc = np.stack([music + beat, np.roll(music, 300) + beat * 0.8], 1)
    if a.mono:
        v, acc = v.mean(1, keepdims=True), acc.mean(1, keepdims=True)
    g = 0.9 / max(1e-9, np.abs(v + acc).max())
    v, acc = (v * g).astype(np.float32), (acc * g).astype(np.float32)
    sf.write(a.out, v + acc, SR, subtype="FLOAT")
    if a.stems:
        sf.write(a.stems + "_vocals.wav", v, SR, subtype="FLOAT")
        sf.write(a.stems + "_instrumental.wav", acc, SR, subtype="FLOAT")


if __name__ == "__main__":
    main()
