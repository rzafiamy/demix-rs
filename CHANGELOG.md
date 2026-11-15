# Changelog

All notable changes to **demix-rs** are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-11-15

### Added
- BS-RoFormer inference in Rust/Candle (CUDA, Metal, CPU): band split,
  time / frequency transformers with interleaved rotary embeddings and
  per-head gates, mask estimators; MSST `demix` chunking (reflect padding,
  linear fades, overlap-add); torch-compatible STFT / iSTFT.
- `demix convert`: MSST checkpoint (`.ckpt` / `.pth` / `.safetensors`) +
  YAML config → one GGUF (F16, BF16, F32 or Q8_0 linear weights); the
  checkpoint is checked against the config before writing.
- `demix separate`: files in (WAV, FLAC, MP3, Ogg, M4A), `<name>_<stem>.wav`
  out; derived `instrumental` (or `vocals`) stem.
- `demix serve`: `POST /v1/audio/separations` (one stem → WAV, several →
  zip), `/v1/models`, `/health`.
- `--attn-budget-mb`: transformers run in slices under a memory budget, so a
  separation fits next to a large LLM on the same GPU.
- Parity tooling: `scripts/ref/ref.py`, `examples/parity.rs`; `tests/tiny.rs`
  (random tiny model, no download).
