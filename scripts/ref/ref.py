#!/usr/bin/env python3
"""BS-RoFormer Python reference: separates a file the way
Music-Source-Separation-Training's `demix` does (generic mode, f32, CPU) and
dumps the result, plus the intermediates of the first chunk, for the parity
checks of the Rust port.

usage: ref.py <model.ckpt> <config.yaml> <input.wav> <out.safetensors> [--no-chunk-dump]

The input must already be 44.1 kHz (stereo or mono); resampling is not part
of the reference. Model code: bs_roformer.py / attend.py next to this file
(MIT, ZFTurbo/Music-Source-Separation-Training, see LICENSE.MSST).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from bs_roformer import BSRoformer


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def window_array(size, fade):
    w = torch.ones(size)
    w[-fade:] = torch.linspace(1, 0, fade)
    w[:fade] = torch.linspace(0, 1, fade)
    return w


def demix(model, mix, chunk_size, num_overlap, batch_size):
    """MSST `demix`, generic mode, without AMP (f32)."""
    fade = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    win = window_array(chunk_size, fade)
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")
    result = None
    counter = torch.zeros(mix.shape[-1])
    i = 0
    batch, locs = [], []
    while i < mix.shape[1]:
        part = mix[:, i : i + chunk_size]
        n = part.shape[-1]
        mode = "reflect" if n > chunk_size // 2 else "constant"
        part = nn.functional.pad(part, (0, chunk_size - n), mode=mode, value=0)
        batch.append(part)
        locs.append((i, n))
        i += step
        if len(batch) >= batch_size or i >= mix.shape[1]:
            x = model(torch.stack(batch, 0))
            if result is None:
                result = torch.zeros((x.shape[1],) + mix.shape)
            for j, (start, seg) in enumerate(locs):
                first = start == 0
                last = start + step >= mix.shape[1]
                w = win.clone()
                if first:
                    w[:fade] = 1
                if last:
                    w[-fade:] = 1
                result[..., start : start + seg] += x[j, ..., :seg] * w[:seg]
                counter[start : start + seg] += w[:seg]
            batch.clear()
            locs.clear()
    out = result / counter
    if length_init > 2 * border and border > 0:
        out = out[..., border:-border]
    return torch.nan_to_num(out, nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("config")
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--no-chunk-dump", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    with open(a.config) as f:
        cfg = yaml.load(f, Loader=Loader)
    mcfg = dict(cfg["model"])
    mcfg["flash_attn"] = False  # einsum path: same maths, no CUDA-only context manager
    model = BSRoformer(**mcfg)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()

    audio, sr = sf.read(a.input, dtype="float32", always_2d=True)
    assert sr == cfg["audio"]["sample_rate"], f"input must be {cfg['audio']['sample_rate']} Hz"
    mix = torch.from_numpy(audio.T.copy())
    if mix.shape[0] == 1 and mcfg.get("stereo", False):
        mix = torch.cat([mix, mix], 0)

    inf = cfg.get("inference", {})
    chunk = inf.get("chunk_size", cfg["audio"]["chunk_size"])
    out = {"mix": mix.contiguous()}

    with torch.no_grad():
        if not a.no_chunk_dump:
            dumps = {}

            def hook(name):
                def f(_m, _i, o):
                    dumps[name] = o.detach().clone()
                return f

            model.band_split.register_forward_hook(hook("band_split"))
            model.layers[0][0].register_forward_hook(hook("layer0_time"))
            model.layers[0][1].register_forward_hook(hook("layer0_freq"))
            model.final_norm.register_forward_hook(hook("final_norm"))
            model.mask_estimators[0].register_forward_hook(hook("mask0"))
            part = mix[:, :chunk]
            part = nn.functional.pad(part, (0, chunk - part.shape[-1]))
            y = model(part[None])
            out["chunk_in"] = part.contiguous()
            out["chunk_out"] = y[0].contiguous()
            for k, v in dumps.items():
                out[k] = v.contiguous()

        est = demix(model, mix, chunk, inf.get("num_overlap", 2), inf.get("batch_size", 1))
    out["sources"] = est.contiguous()
    save_file(out, a.out)
    print({k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
