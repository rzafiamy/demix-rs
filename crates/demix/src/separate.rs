//! Whole-track separation: Music-Source-Separation-Training's `demix`
//! (generic mode): reflect-pad the track by `chunk - step`, run overlapping
//! chunks (step = chunk / num_overlap) with linear fades of chunk / 10
//! (none at the track's two ends), average by the summed fades, unpad.
//!
//! Per chunk: STFT of each channel -> model -> complex masks -> masked
//! spectrum (DC zeroed when `zero_dc`) -> iSTFT.

use crate::config::Config;
use crate::model::BsRoformer;
use crate::stft::{Spec, Stft};
use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use realfft::num_complex::Complex32;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Planar audio: `[channel][sample]`.
pub type Planar = Vec<Vec<f32>>;

pub struct Separator {
    pub cfg: Config,
    pub model: BsRoformer,
    stft: Stft,
    /// Chunks per model call.
    pub batch_size: usize,
    /// Budget an OOM retry settled on (0: none yet); later chunks start there.
    settled_budget: AtomicUsize,
}

/// One named output track.
pub struct Stem {
    pub name: String,
    pub audio: Planar,
}

impl Separator {
    pub fn new(cfg: Config, model: BsRoformer) -> Self {
        let stft = Stft::new(cfg.stft_n_fft, cfg.stft_hop_length, cfg.stft_normalized);
        Self {
            cfg,
            model,
            stft,
            batch_size: 1,
            settled_budget: AtomicUsize::new(0),
        }
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }

    /// Model input `[t, bins * channels * 2]` for one chunk, and the spectra.
    fn features(&self, chunk: &Planar) -> Result<(Vec<f32>, Vec<Spec>)> {
        let specs: Vec<Spec> = chunk
            .iter()
            .map(|c| self.stft.forward(c))
            .collect::<Result<_>>()?;
        let ch = specs.len();
        let (bins, frames) = (specs[0].bins, specs[0].frames);
        let width = bins * ch * 2;
        let mut feat = vec![0f32; frames * width];
        for t in 0..frames {
            let row = &mut feat[t * width..(t + 1) * width];
            for f in 0..bins {
                for (s, spec) in specs.iter().enumerate() {
                    let z = spec.data[t * bins + f];
                    let o = (f * ch + s) * 2;
                    row[o] = z.re;
                    row[o + 1] = z.im;
                }
            }
        }
        Ok((feat, specs))
    }

    /// The model forward; on a GPU out-of-memory error (another process may
    /// hold most of the GPU for a while) it retries with a budget divided by
    /// 4 each time, down to 16 MiB.
    fn forward_retrying(&self, x: &Tensor) -> Result<Tensor> {
        let mut budget = match self.settled_budget.load(Ordering::Relaxed) {
            0 => self.model.attn_budget,
            b => b.min(self.model.attn_budget),
        };
        loop {
            match self.model.forward_budget(x, budget) {
                Ok(m) => {
                    if budget < self.model.attn_budget {
                        self.settled_budget.store(budget, Ordering::Relaxed);
                    }
                    return Ok(m);
                }
                Err(e) if is_oom(&e) && budget > MIN_BUDGET => {
                    budget = (budget / 4).max(MIN_BUDGET);
                    tracing::warn!(
                        "GPU out of memory: retrying with {} MiB slices",
                        budget >> 20
                    );
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Runs a batch of chunks (each `[channel][chunk_size]`); returns
    /// `[chunk][stem]` planar audio of `chunk_size` samples.
    pub fn run_chunks(&self, chunks: &[Planar]) -> Result<Vec<Vec<Planar>>> {
        let ch = self.cfg.channels();
        let mut feats = Vec::with_capacity(chunks.len());
        let mut all_specs = Vec::with_capacity(chunks.len());
        for c in chunks {
            if c.len() != ch {
                bail!("chunk has {} channels, the model takes {ch}", c.len());
            }
            let (f, s) = self.features(c)?;
            feats.push(f);
            all_specs.push(s);
        }
        let frames = all_specs[0][0].frames;
        let bins = all_specs[0][0].bins;
        let width = bins * ch * 2;
        let flat: Vec<f32> = feats.concat();
        let x = Tensor::from_vec(flat, (chunks.len(), frames, width), &Device::Cpu)?
            .to_device(self.device())?;
        let masks = self.forward_retrying(&x)?.to_device(&Device::Cpu)?; // [b, n, t, w]
        let (_, stems, _, _) = masks.dims4()?;
        let masks: Vec<f32> = masks.flatten_all()?.to_vec1()?;

        let len = chunks[0][0].len();
        let mut out = Vec::with_capacity(chunks.len());
        for (bi, specs) in all_specs.iter().enumerate() {
            let mut per_stem = Vec::with_capacity(stems);
            for n in 0..stems {
                let base = (bi * stems + n) * frames * width;
                let mut planar = Vec::with_capacity(ch);
                for (s, spec) in specs.iter().enumerate() {
                    let mut data = vec![Complex32::new(0.0, 0.0); frames * bins];
                    for t in 0..frames {
                        for f in 0..bins {
                            let o = base + t * width + (f * ch + s) * 2;
                            let m = Complex32::new(masks[o], masks[o + 1]);
                            data[t * bins + f] = spec.data[t * bins + f] * m;
                        }
                        if self.cfg.zero_dc {
                            data[t * bins] = Complex32::new(0.0, 0.0);
                        }
                    }
                    let masked = Spec { bins, frames, data };
                    planar.push(self.stft.inverse(&masked, len)?);
                }
                per_stem.push(planar);
            }
            out.push(per_stem);
        }
        Ok(out)
    }

    /// Separates a whole track (`[channel][sample]` at the model's sample
    /// rate and channel count). `progress(done, total)` is called after each
    /// batch. Returns the model's stems, then the derived one
    /// (`instrumental` = mix - vocals, or `vocals` = mix - instrumental).
    pub fn separate(
        &self,
        mix: &Planar,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Vec<Stem>> {
        let ch = self.cfg.channels();
        if mix.len() != ch {
            bail!("audio has {} channels, the model takes {ch}", mix.len());
        }
        let length_init = mix[0].len();
        if length_init == 0 {
            bail!("empty audio");
        }
        let chunk = self.cfg.chunk_size;
        let step = chunk / self.cfg.num_overlap;
        let border = chunk - step;
        let fade = chunk / 10;
        let padded = length_init > 2 * border && border > 0;
        let work: Planar = if padded {
            mix.iter().map(|c| reflect_pad(c, border, border)).collect()
        } else {
            mix.clone()
        };
        let total = work[0].len();
        let window = fade_window(chunk, fade);

        let stems = self.cfg.num_stems;
        let mut result = vec![vec![vec![0f32; total]; ch]; stems];
        let mut counter = vec![0f32; total];

        let starts: Vec<usize> = (0..total).step_by(step).collect();
        let mut done = 0;
        for group in starts.chunks(self.batch_size.max(1)) {
            let mut batch = Vec::with_capacity(group.len());
            for &start in group {
                let n = chunk.min(total - start);
                let part: Planar = work
                    .iter()
                    .map(|c| {
                        let seg = &c[start..start + n];
                        if n > chunk / 2 {
                            reflect_pad(seg, 0, chunk - n)
                        } else {
                            let mut v = seg.to_vec();
                            v.resize(chunk, 0.0);
                            v
                        }
                    })
                    .collect();
                batch.push(part);
            }
            let outs = self.run_chunks(&batch)?;
            for (&start, est) in group.iter().zip(outs) {
                let n = chunk.min(total - start);
                let first = start == 0;
                let last = start + step >= total;
                let mut w = window.clone();
                if fade > 0 && first {
                    w[..fade].fill(1.0);
                }
                if fade > 0 && last {
                    w[chunk - fade..].fill(1.0);
                }
                for (si, planar) in est.iter().enumerate() {
                    for (c, samples) in planar.iter().enumerate() {
                        let dst = &mut result[si][c][start..start + n];
                        for k in 0..n {
                            dst[k] += samples[k] * w[k];
                        }
                    }
                }
                for k in 0..n {
                    counter[start + k] += w[k];
                }
            }
            done += group.len();
            progress(done, starts.len());
        }

        let (lo, hi) = if padded {
            (border, total - border)
        } else {
            (0, total)
        };
        let mut out: Vec<Stem> = result
            .into_iter()
            .zip(&self.cfg.stems)
            .map(|(planar, name)| Stem {
                name: name.clone(),
                audio: planar
                    .into_iter()
                    .map(|c| {
                        c[lo..hi]
                            .iter()
                            .zip(&counter[lo..hi])
                            .map(|(v, n)| {
                                let y = v / n;
                                if y.is_nan() { 0.0 } else { y }
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect();

        let has = |n: &str| out.iter().any(|s| s.name == n);
        let derived = if has("vocals") && !has("instrumental") {
            Some(("vocals", "instrumental"))
        } else if has("instrumental") && !has("vocals") {
            Some(("instrumental", "vocals"))
        } else {
            None
        };
        if let Some((from, name)) = derived {
            let src = out.iter().find(|s| s.name == from).unwrap();
            let audio = mix
                .iter()
                .zip(&src.audio)
                .map(|(m, s)| m.iter().zip(s).map(|(a, b)| a - b).collect())
                .collect();
            out.push(Stem {
                name: name.into(),
                audio,
            });
        }
        Ok(out)
    }

    /// Names `separate` returns, in order.
    pub fn stem_names(&self) -> Vec<String> {
        let mut v = self.cfg.stems.clone();
        let has = |n: &str| v.iter().any(|s| s == n);
        if has("vocals") && !has("instrumental") {
            v.push("instrumental".into());
        } else if has("instrumental") && !has("vocals") {
            v.push("vocals".into());
        }
        v
    }
}

/// Smallest activation budget the OOM retries go down to.
const MIN_BUDGET: usize = 16 << 20;

fn is_oom(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    s.contains("OUT_OF_MEMORY") || s.contains("out of memory")
}

/// `torch.nn.functional.pad(x, (left, right), mode="reflect")`.
pub fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    assert!(
        n > 1 && left < n && right < n,
        "reflect pad larger than the signal"
    );
    let mut v = Vec::with_capacity(left + n + right);
    for i in (1..=left).rev() {
        v.push(x[i]);
    }
    v.extend_from_slice(x);
    for i in 0..right {
        v.push(x[n - 2 - i]);
    }
    v
}

/// MSST `_getWindowingArray`: linear fade in / out of `fade` samples.
pub fn fade_window(size: usize, fade: usize) -> Vec<f32> {
    let mut w = vec![1f32; size];
    if fade == 0 {
        return w;
    }
    let lin = |i: usize| {
        if fade == 1 {
            0.0
        } else {
            i as f32 / (fade - 1) as f32
        }
    };
    for (i, v) in w[size - fade..].iter_mut().enumerate() {
        *v = 1.0 - lin(i);
    }
    for (i, v) in w[..fade].iter_mut().enumerate() {
        *v = lin(i);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_matches_torch() {
        assert_eq!(
            reflect_pad(&[1., 2., 3., 4.], 2, 3),
            vec![3., 2., 1., 2., 3., 4., 3., 2., 1.]
        );
    }

    #[test]
    fn window_matches_msst_example() {
        let w = fade_window(10, 3);
        assert_eq!(w, vec![0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]);
    }
}
