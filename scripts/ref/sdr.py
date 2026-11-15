#!/usr/bin/env python3
"""SDR (dB) of estimated stems against references, and between two runs.

usage: sdr.py <ref.wav> <est.wav> [<ref2.wav> <est2.wav> ...]
"""
import sys

import numpy as np
import soundfile as sf


def sdr(ref, est):
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return 10 * np.log10((ref ** 2).sum() / max(1e-20, ((ref - est) ** 2).sum()))


args = sys.argv[1:]
for r, e in zip(args[::2], args[1::2]):
    a, _ = sf.read(r, dtype="float64", always_2d=True)
    b, _ = sf.read(e, dtype="float64", always_2d=True)
    print(f"{sdr(a, b):7.2f} dB  {e}  vs  {r}")
