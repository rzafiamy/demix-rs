# Credits

- **BS-RoFormer**: Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, Yun-Ning
  Hung, "Music Source Separation with Band-Split RoPE Transformer"
  (ICASSP 2024, arXiv:2309.02612).
- **lucidrains/BS-RoFormer** (MIT): the reference PyTorch implementation.
- **ZFTurbo/Music-Source-Separation-Training** (MIT, Roman Solovyev):
  model code, configs and the `demix` chunking this port follows.
  `scripts/ref/bs_roformer.py` and `attend.py` are copied from it, unchanged
  except for one import line, for the parity reference only
  (`scripts/ref/LICENSE.MSST`).
- **rotary-embedding-torch** (MIT, lucidrains): rotary embedding semantics.
- **Candle** (Apache-2.0 / MIT, Hugging Face): tensors, CUDA/Metal kernels,
  GGUF and pickle readers.
- **symphonia**, **rubato**, **hound**, **realfft**: audio decoding,
  resampling, WAV writing, FFT.

## Model weights

Weights are not part of this repository and keep their own licences; check
the model card before any commercial use. Tested:

- `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (viperx), published in the UVR
  model repository (TRvlvr/model_repo), config from MSST
  `configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml`.
