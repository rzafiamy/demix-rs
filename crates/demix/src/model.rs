//! BS-RoFormer (Lu et al. 2023), as implemented by lucidrains/BS-RoFormer and
//! Music-Source-Separation-Training (`models/bs_roformer/bs_roformer.py`):
//! band split -> `depth` x (time transformer, frequency transformer) ->
//! RMSNorm -> one mask estimator per stem.
//!
//! The model maps STFT features `[batch, frames, bins * channels * 2]`
//! (order: bin, channel, re/im) to complex masks of the same layout; the STFT,
//! the mask product and the iSTFT are in `separate.rs`.

use crate::config::Config;
use crate::weights::{Linear, RmsNorm, Weights};
use anyhow::{Result, bail};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};

/// Bytes the attention scores of one call may take; the batch is split to
/// stay below it.
pub const DEFAULT_ATTN_BUDGET: usize = 1 << 30;

struct Rotary {
    /// `[dim_head / 2]` F32 frequencies.
    freqs: Tensor,
}

impl Rotary {
    fn load(w: &Weights, name: &str, cfg: &Config) -> Result<Self> {
        let freqs = if w.contains(name) {
            w.get(name, DType::F32)?
        } else {
            let half = cfg.dim_head / 2;
            let v: Vec<f32> = (0..half)
                .map(|i| (1.0 / cfg.rope_theta.powf((2 * i) as f64 / cfg.dim_head as f64)) as f32)
                .collect();
            Tensor::from_vec(v, half, &w.device)?
        };
        Ok(Self { freqs })
    }

    /// cos, sin of `[seq, dim_head / 2]`.
    fn cos_sin(&self, seq: usize) -> Result<(Tensor, Tensor)> {
        let pos = Tensor::arange(0u32, seq as u32, self.freqs.device())?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let ang = pos.broadcast_mul(&self.freqs.unsqueeze(0)?)?;
        Ok((ang.cos()?, ang.sin()?))
    }
}

struct Attention {
    norm: RmsNorm,
    to_qkv: Linear,
    to_gates: Linear,
    to_out: Linear,
    heads: usize,
    dim_head: usize,
}

impl Attention {
    fn load(w: &Weights, p: &str, cfg: &Config, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(w, &format!("{p}.norm"))?,
            to_qkv: w.linear(&format!("{p}.to_qkv"), false, dtype)?,
            to_gates: w.linear(&format!("{p}.to_gates"), true, dtype)?,
            to_out: w.linear(&format!("{p}.to_out.0"), false, dtype)?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
        })
    }

    /// `x`: `[n, seq, dim]`.
    fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), budget: usize) -> Result<Tensor> {
        let (n, s, _) = x.dims3()?;
        let (h, dh) = (self.heads, self.dim_head);
        let xn = self.norm.forward(x)?;
        let qkv = self
            .to_qkv
            .forward(&xn)?
            .reshape((n, s, 3, h, dh))?
            .permute((2, 0, 3, 1, 4))?; // [3, n, h, s, dh]
        // rotary in F32, as rotary_embedding_torch (autocast disabled)
        let q = qkv.i(0)?.to_dtype(DType::F32)?.contiguous()?;
        let k = qkv.i(1)?.to_dtype(DType::F32)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope_i(&q, &rope.0, &rope.1)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &rope.0, &rope.1)?;
        let dtype = v.dtype();
        let q = (q * (dh as f64).powf(-0.5))?.to_dtype(dtype)?;
        let k = k.to_dtype(dtype)?;

        // scores [n, h, s, s] in F32: split n to keep them under `budget`
        let per = h * s * s * 4;
        let step = (budget / per.max(1)).clamp(1, n);
        let mut outs = Vec::with_capacity(n.div_ceil(step));
        let mut i = 0;
        while i < n {
            let m = step.min(n - i);
            let qs = q.narrow(0, i, m)?;
            let ks = k.narrow(0, i, m)?;
            let vs = v.narrow(0, i, m)?;
            let sim = qs.matmul(&ks.t()?.contiguous()?)?.to_dtype(DType::F32)?;
            let attn = candle_nn::ops::softmax_last_dim(&sim)?.to_dtype(dtype)?;
            outs.push(attn.matmul(&vs)?);
            i += m;
        }
        let out = Tensor::cat(&outs, 0)?; // [n, h, s, dh]

        let gates = candle_nn::ops::sigmoid(&self.to_gates.forward(&xn)?)?
            .permute((0, 2, 1))?
            .unsqueeze(3)?; // [n, h, s, 1]
        let out = out.broadcast_mul(&gates.to_dtype(out.dtype())?)?;
        let out = out.permute((0, 2, 1, 3))?.reshape((n, s, h * dh))?;
        Ok(self.to_out.forward(&out)?)
    }
}

struct FeedForward {
    norm: RmsNorm,
    lin1: Linear,
    lin2: Linear,
}

impl FeedForward {
    fn load(w: &Weights, p: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(w, &format!("{p}.net.0"))?,
            lin1: w.linear(&format!("{p}.net.1"), true, dtype)?,
            lin2: w.linear(&format!("{p}.net.4"), true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.lin1.forward(&self.norm.forward(x)?)?;
        Ok(self.lin2.forward(&y.gelu_erf()?)?)
    }
}

struct Transformer {
    layers: Vec<(Attention, FeedForward)>,
    rotary: Rotary,
}

impl Transformer {
    fn load(w: &Weights, p: &str, depth: usize, cfg: &Config, dtype: DType) -> Result<Self> {
        let mut layers = Vec::with_capacity(depth);
        for k in 0..depth {
            layers.push((
                Attention::load(w, &format!("{p}.layers.{k}.0"), cfg, dtype)?,
                FeedForward::load(w, &format!("{p}.layers.{k}.1"), dtype)?,
            ));
        }
        let rotary = Rotary::load(w, &format!("{p}.layers.0.0.rotary_embed.freqs"), cfg)?;
        Ok(Self { layers, rotary })
    }

    /// `x`: `[n, seq, dim]`; rows are independent, so they run in slices
    /// whose activations stay under `budget` bytes.
    fn forward(&self, x: &Tensor, budget: usize) -> Result<Tensor> {
        let (n, s, d) = x.dims3()?;
        let rope = self.rotary.cos_sin(s)?;
        // ~24 F32-sized copies of a row live at once (qkv, F32 q/k, rotary,
        // the feed-forward's 4x hidden and its GELU), plus attention scores
        let heads = self.layers.first().map_or(1, |l| l.0.heads);
        let per_row = s * d * 4 * 24 + heads * s * s * 4;
        let step = (budget / per_row.max(1)).clamp(1, n);
        let run = |x: &Tensor| -> Result<Tensor> {
            let mut x = x.clone();
            for (attn, ff) in &self.layers {
                x = (attn.forward(&x, &rope, budget)? + &x)?;
                x = (ff.forward(&x)? + &x)?;
            }
            Ok(x)
        };
        if step >= n {
            return run(x);
        }
        let mut outs = Vec::with_capacity(n.div_ceil(step));
        let mut i = 0;
        while i < n {
            let m = step.min(n - i);
            outs.push(run(&x.narrow(0, i, m)?)?);
            i += m;
        }
        Ok(Tensor::cat(&outs, 0)?)
    }
}

struct BandSplit {
    bands: Vec<(RmsNorm, Linear)>,
    dims: Vec<usize>,
}

impl BandSplit {
    /// `x`: `[b, t, sum(dims)]` -> `[b, t, bands, dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outs = Vec::with_capacity(self.bands.len());
        let mut off = 0;
        for ((norm, lin), &d) in self.bands.iter().zip(&self.dims) {
            let part = x.narrow(D::Minus1, off, d)?;
            outs.push(lin.forward(&norm.forward(&part)?)?);
            off += d;
        }
        Ok(Tensor::stack(&outs, 2)?)
    }
}

struct MaskEstimator {
    /// Per band: the MLP's linear layers (tanh between them), then GLU.
    bands: Vec<Vec<Linear>>,
}

impl MaskEstimator {
    /// `x`: `[b, t, bands, dim]` -> `[b, t, sum(dims)]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outs = Vec::with_capacity(self.bands.len());
        for (i, mlp) in self.bands.iter().enumerate() {
            let mut y = x.i((.., .., i))?;
            for (j, lin) in mlp.iter().enumerate() {
                if j > 0 {
                    y = y.tanh()?;
                }
                y = lin.forward(&y)?;
            }
            let half = y.dim(D::Minus1)? / 2;
            let a = y.narrow(D::Minus1, 0, half)?;
            let g = y.narrow(D::Minus1, half, half)?;
            outs.push((a * candle_nn::ops::sigmoid(&g)?)?);
        }
        Ok(Tensor::cat(&outs, D::Minus1)?)
    }
}

pub struct BsRoformer {
    band_split: BandSplit,
    layers: Vec<(Transformer, Transformer)>,
    final_norm: RmsNorm,
    mask_estimators: Vec<MaskEstimator>,
    skip_connection: bool,
    pub dtype: DType,
    pub device: Device,
    pub attn_budget: usize,
}

impl BsRoformer {
    pub fn load(w: &Weights, cfg: &Config, dtype: DType) -> Result<Self> {
        cfg.validate()?;
        let dims = cfg.band_dims();
        let mut bands = Vec::with_capacity(dims.len());
        for i in 0..dims.len() {
            let p = format!("band_split.to_features.{i}");
            bands.push((
                RmsNorm::load(w, &format!("{p}.0"))?,
                w.linear(&format!("{p}.1"), true, dtype)?,
            ));
        }
        let band_split = BandSplit {
            bands,
            dims: dims.clone(),
        };

        let mut layers = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            layers.push((
                Transformer::load(
                    w,
                    &format!("layers.{i}.0"),
                    cfg.time_transformer_depth,
                    cfg,
                    dtype,
                )?,
                Transformer::load(
                    w,
                    &format!("layers.{i}.1"),
                    cfg.freq_transformer_depth,
                    cfg,
                    dtype,
                )?,
            ));
        }

        let mut mask_estimators = Vec::with_capacity(cfg.num_stems);
        for n in 0..cfg.num_stems {
            let mut per_band = Vec::with_capacity(dims.len());
            for b in 0..dims.len() {
                let mut mlp = Vec::with_capacity(cfg.mask_estimator_depth);
                for j in 0..cfg.mask_estimator_depth {
                    // Sequential(MLP, GLU): the MLP's linears sit at even indices
                    let p = format!("mask_estimators.{n}.to_freqs.{b}.0.{}", 2 * j);
                    mlp.push(w.linear(&p, true, dtype)?);
                }
                per_band.push(mlp);
            }
            mask_estimators.push(MaskEstimator { bands: per_band });
        }
        if w.contains(&format!(
            "mask_estimators.{}.to_freqs.0.0.0.weight",
            cfg.num_stems
        )) {
            bail!(
                "checkpoint has more mask estimators than num_stems = {}",
                cfg.num_stems
            );
        }

        Ok(Self {
            band_split,
            layers,
            final_norm: RmsNorm::load(w, "final_norm")?,
            mask_estimators,
            skip_connection: cfg.skip_connection,
            dtype,
            device: w.device.clone(),
            attn_budget: DEFAULT_ATTN_BUDGET,
        })
    }

    /// `x`: `[b, t, features]` -> masks `[b, stems, t, features]` (F32).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_budget(x, self.attn_budget)
    }

    /// [`forward`](Self::forward) with an explicit activation budget (bytes).
    pub fn forward_budget(&self, x: &Tensor, budget: usize) -> Result<Tensor> {
        let x = x.to_dtype(self.dtype)?;
        let mut x = self.band_split.forward(&x)?; // [b, t, f, d]
        let (b, t, f, d) = x.dims4()?;
        let mut store: Vec<Tensor> = Vec::new();
        for (time, freq) in &self.layers {
            if self.skip_connection {
                for s in &store {
                    x = (x + s)?;
                }
            }
            let y = x.permute((0, 2, 1, 3))?.reshape((b * f, t, d))?;
            let y = time.forward(&y, budget)?;
            let y = y
                .reshape((b, f, t, d))?
                .permute((0, 2, 1, 3))?
                .reshape((b * t, f, d))?;
            let y = freq.forward(&y, budget)?;
            x = y.reshape((b, t, f, d))?;
            if self.skip_connection {
                store.push(x.clone());
            }
        }
        let x = self.final_norm.forward(&x)?;
        let masks: Vec<Tensor> = self
            .mask_estimators
            .iter()
            .map(|m| m.forward(&x))
            .collect::<Result<_>>()?;
        Ok(Tensor::stack(&masks, 1)?.to_dtype(DType::F32)?)
    }

    /// Intermediates of one forward for the parity checks: band split,
    /// first time / frequency transformer, final norm, masks.
    pub fn forward_debug(&self, x: &Tensor) -> Result<Vec<(&'static str, Tensor)>> {
        let x = x.to_dtype(self.dtype)?;
        let mut out = Vec::new();
        let mut x = self.band_split.forward(&x)?;
        out.push(("band_split", x.clone()));
        let (b, t, f, d) = x.dims4()?;
        for (i, (time, freq)) in self.layers.iter().enumerate() {
            let y = x.permute((0, 2, 1, 3))?.reshape((b * f, t, d))?;
            let y = time.forward(&y, self.attn_budget)?;
            if i == 0 {
                out.push(("layer0_time", y.clone()));
            }
            let y = y
                .reshape((b, f, t, d))?
                .permute((0, 2, 1, 3))?
                .reshape((b * t, f, d))?;
            let y = freq.forward(&y, self.attn_budget)?;
            if i == 0 {
                out.push(("layer0_freq", y.clone()));
            }
            x = y.reshape((b, t, f, d))?;
        }
        let x = self.final_norm.forward(&x)?;
        out.push(("final_norm", x.clone()));
        out.push(("mask0", self.mask_estimators[0].forward(&x)?));
        Ok(out)
    }
}
