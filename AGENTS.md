# AGENTS.md

Guidance for AI agents working in this repository.

## Project

Rust/Candle port of BS-RoFormer music source separation (inference only),
one binary (`demix`), models in one GGUF written by `demix convert` from a
Music-Source-Separation-Training (MSST) checkpoint + YAML config. Engine of
zallama's `demix-server` backend (`POST /v1/audio/separations`).

Pipeline: decode → conform (stereo, 44.1 kHz) → MSST `demix` chunking
(`separate.rs`: reflect pad, 8 s chunks, step chunk / num_overlap, linear
fades) → per chunk STFT (`stft.rs`, torch.stft semantics) → model
(`model.rs`: band split → depth × (time transformer, frequency
transformer) → RMSNorm → mask estimators) → complex mask × spectrum, DC
zeroed → iSTFT → overlap-add.

## Where things are

- `crates/demix/src/config.rs`: MSST YAML → `Config`, stored as JSON in the GGUF.
- `crates/demix/src/model.rs`: the network; `forward_debug` for parity dumps.
- `crates/demix/src/weights.rs`: dense / `QMatMul` linear, RMSNorm
  (`F.normalize * sqrt(dim) * gamma`, not the eps-RMS kind).
- `crates/demix/src/gguf.rs`: checkpoint reading (candle pickle, tensors
  only), conversion, loading.
- `crates/demix/src/audio.rs`: symphonia decode, rubato resample, hound WAV.
- `crates/demix-cli/src/`: CLI (`main.rs`), HTTP server (`server.rs`).
- `scripts/ref/`: Python reference (`ref.py`, vendored MSST model code),
  synthetic test audio (`make_test_audio.py`), `sdr.py`.

## Commands

```bash
cargo build --release -p demix-cli
cargo test --release --workspace
cargo clippy --workspace --all-targets -- -D warnings
demix convert <model.ckpt> <config.yaml> -o models/x.gguf [--dtype f16|q8_0]
demix separate -m models/x.gguf song.mp3 -o out/

# parity (Python venv with torch, einops, rotary-embedding-torch, beartype, pyyaml, soundfile, safetensors)
python scripts/ref/make_test_audio.py <speech.wav> tmp/mix20.wav --stems tmp/true20
python scripts/ref/ref.py <model.ckpt> <config.yaml> tmp/mix20.wav tmp/ref.safetensors
cargo run --release --example parity -- <model.ckpt> <config.yaml> tmp/ref.safetensors --cpu
```

CUDA: `--features cuda`, build into `--target-dir target-cuda`, needs
`/usr/local/cuda/bin` in PATH and `CUDA_COMPUTE_CAP`.

## Rules

- MSST's `models/bs_roformer/bs_roformer.py` and `utils/model_utils.py`
  (`demix`) are the spec. Check model changes with the parity example: F32
  CPU must match the reference (SNR well above 60 dB on every dump).
- Rotary embeddings are interleaved (`rope_i`), positions from 0 on each
  axis, frequencies from the checkpoint.
- candle 0.11 pitfalls: `matmul` with a stride-0 batch (`broadcast_left`)
  returns wrong values — flatten to 2-D (see `weights.rs`).
- Keep the CPU path dependency-free (no ffmpeg, no system libs).
- Tests need no model download (`tests/tiny.rs` builds a random one).
