# demix-rs

Music source separation in Rust: **BS-RoFormer** (band-split RoFormer,
Lu et al. 2023) on [Candle](https://github.com/huggingface/candle), one GGUF
file per model. Song in, vocals and instrumental out: karaoke tracks, backing
tracks for a choir, a cappella stems.

- One binary, `demix`: `convert`, `separate`, `serve` (HTTP).
- Pure Rust audio: decodes WAV, FLAC, MP3, Ogg Vorbis, AAC/ALAC (MP4/M4A),
  resamples to the model's rate, writes WAV. No ffmpeg, no Python.
- CUDA, Metal or CPU. F16 / BF16 / F32 compute; F16 or Q8_0 weights.
- Runs any Music-Source-Separation-Training BS-RoFormer checkpoint (viperx,
  anvuew, becruily, unwa, Gabox fine-tunes…) from its `.ckpt` and YAML
  config. Mel-Band RoFormer is not supported yet.
- The engine behind zallama's `separation` modality
  ([rzafiamy/zallamars](https://github.com/rzafiamy/zallamars), backend
  `demix-server`).

## Build

```bash
./build.sh --cuda      # NVIDIA (needs nvcc), or --metal, or nothing for CPU
```

## Models

Convert a checkpoint and its MSST config (the checkpoint is read as data: no
code from it runs):

```bash
demix convert model_bs_roformer_ep_317_sdr_12.9755.ckpt \
  model_bs_roformer_ep_317_sdr_12.9755.yaml \
  -o models/bs-roformer-viperx-317-q8_0.gguf --dtype q8_0
```

| dtype | viperx 317 file | note |
|---|---|---|
| `f16` (default) | 320 MB | |
| `q8_0` | 171 MB | output within ~50 dB of F32; fits next to a large LLM |

Check each model's licence before use: many community checkpoints are
non-commercial.

## Separate

```bash
demix separate -m models/bs-roformer-viperx-317-q8_0.gguf song.mp3 -o out/
# out/song_vocals.wav, out/song_instrumental.wav
demix separate -m … --stems instrumental --format f32 *.flac -o karaoke/
```

Options: `--dtype f16|bf16|f32`, `--batch N` (chunks per model call),
`--overlap N` (passes per sample, model default 2; 4 is a little cleaner and
twice as slow), `--attn-budget-mb N` (activation memory per transformer
slice, default 1024: lower it on a busy GPU), `--cpu`, `--threads N`.

A vocals model also outputs `instrumental` (mix − vocals); an instrumental
model also outputs `vocals`.

## HTTP API

```bash
demix serve -m models/bs-roformer-viperx-317-q8_0.gguf --port 8790
curl -F file=@chant.mp3 -F stem=instrumental http://127.0.0.1:8790/v1/audio/separations -o instru.wav
curl -F file=@chant.mp3 http://127.0.0.1:8790/v1/audio/separations -o stems.zip
```

`POST /v1/audio/separations` (multipart): `file`, optional `stem` (one name,
several comma-separated, or `all`) and `format` (`s16` | `f32`). One stem →
`audio/wav`, several → `application/zip`. `GET /v1/models` lists the stems;
`GET /health`. Requests run one at a time. Errors:
`{"error": {"message", "type"}}`.

## Accuracy

The Python reference (`scripts/ref/ref.py`, Music-Source-Separation-Training's
model code and `demix`) dumps the first chunk's intermediates and the whole
track; `cargo run --release --example parity` compares them (see
`AGENTS.md`). `tests/tiny.rs` checks checkpoint → GGUF → separation on a tiny
random model without any download.

## License

Apache-2.0 (code). `scripts/ref/bs_roformer.py` and `attend.py` are MIT
(ZFTurbo/Music-Source-Separation-Training), used only as the parity
reference. Model weights carry their own licences. See `CREDITS.md`.
