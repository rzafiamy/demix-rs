//! STFT / iSTFT matching `torch.stft(center=True, pad_mode="reflect",
//! onesided=True, return_complex=True)` and `torch.istft(center=True,
//! length=…)` with a periodic Hann window of `n_fft` samples.

use anyhow::{Result, bail};
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f32>,
    normalized: bool,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
}

/// Spectrum of one channel: `bins x frames`, frame-major
/// (`data[t * bins + f]`).
pub struct Spec {
    pub bins: usize,
    pub frames: usize,
    pub data: Vec<Complex32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize, normalized: bool) -> Self {
        let window = (0..n_fft)
            .map(|n| {
                let x = 2.0 * std::f64::consts::PI * n as f64 / n_fft as f64;
                (0.5 - 0.5 * x.cos()) as f32
            })
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            n_fft,
            hop,
            window,
            normalized,
            fwd: planner.plan_fft_forward(n_fft),
            inv: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    pub fn forward(&self, x: &[f32]) -> Result<Spec> {
        let pad = self.n_fft / 2;
        if x.len() <= pad {
            bail!(
                "signal of {} samples is too short for the STFT (needs > {pad})",
                x.len()
            );
        }
        // reflect padding (no edge repeat), as torch's pad_mode="reflect"
        let n = x.len();
        let at = |i: isize| -> f32 {
            let mut i = i;
            if i < 0 {
                i = -i;
            }
            if i >= n as isize {
                i = 2 * (n as isize - 1) - i;
            }
            x[i as usize]
        };
        let frames = self.frames(n);
        let bins = self.bins();
        let scale = if self.normalized {
            1.0 / (self.n_fft as f32).sqrt()
        } else {
            1.0
        };
        let mut data = vec![Complex32::new(0.0, 0.0); frames * bins];
        let mut inbuf = self.fwd.make_input_vec();
        let mut outbuf = self.fwd.make_output_vec();
        let mut scratch = self.fwd.make_scratch_vec();
        for t in 0..frames {
            let start = (t * self.hop) as isize - pad as isize;
            for (k, v) in inbuf.iter_mut().enumerate() {
                *v = at(start + k as isize) * self.window[k];
            }
            self.fwd
                .process_with_scratch(&mut inbuf, &mut outbuf, &mut scratch)
                .map_err(|e| anyhow::anyhow!("fft: {e}"))?;
            let row = &mut data[t * bins..(t + 1) * bins];
            for (d, s) in row.iter_mut().zip(outbuf.iter()) {
                *d = s * scale;
            }
        }
        Ok(Spec { bins, frames, data })
    }

    /// Inverse STFT, trimmed / zero-padded to `length` samples.
    pub fn inverse(&self, spec: &Spec, length: usize) -> Result<Vec<f32>> {
        if spec.bins != self.bins() {
            bail!("spectrum has {} bins, expected {}", spec.bins, self.bins());
        }
        let n_fft = self.n_fft;
        let total = n_fft + self.hop * (spec.frames.saturating_sub(1));
        let mut y = vec![0f32; total];
        let mut env = vec![0f32; total];
        let mut inbuf = self.inv.make_input_vec();
        let mut outbuf = self.inv.make_output_vec();
        let mut scratch = self.inv.make_scratch_vec();
        // irfft with torch's "backward" norm (1/N), and the undoing of
        // `normalized` (x sqrt(N))
        let scale = if self.normalized {
            1.0 / (n_fft as f32).sqrt()
        } else {
            1.0 / n_fft as f32
        };
        for t in 0..spec.frames {
            inbuf.copy_from_slice(&spec.data[t * spec.bins..(t + 1) * spec.bins]);
            // pocketfft's c2r ignores these imaginary parts too
            inbuf[0].im = 0.0;
            let last = inbuf.len() - 1;
            inbuf[last].im = 0.0;
            self.inv
                .process_with_scratch(&mut inbuf, &mut outbuf, &mut scratch)
                .map_err(|e| anyhow::anyhow!("ifft: {e}"))?;
            let off = t * self.hop;
            for k in 0..n_fft {
                let w = self.window[k];
                y[off + k] += outbuf[k] * scale * w;
                env[off + k] += w * w;
            }
        }
        let start = n_fft / 2;
        let mut out = vec![0f32; length];
        for (i, o) in out.iter_mut().enumerate() {
            let j = start + i;
            if j >= total {
                break;
            }
            if env[j].abs() < 1e-11 {
                bail!("iSTFT window envelope is zero at sample {i}");
            }
            *o = y[j] / env[j];
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_identity() {
        let s = Stft::new(2048, 441, false);
        let x: Vec<f32> = (0..20_000)
            .map(|i| {
                ((i as f32) * 0.013).sin() * 0.5 + ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.1
            })
            .collect();
        let spec = s.forward(&x).unwrap();
        assert_eq!(spec.frames, 1 + 20_000 / 441);
        assert_eq!(spec.bins, 1025);
        let y = s.inverse(&spec, x.len()).unwrap();
        let err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-5, "max err {err}");
    }

    #[test]
    fn dc_bin_of_constant_is_window_sum() {
        let s = Stft::new(16, 4, false);
        let spec = s.forward(&[1.0; 64]).unwrap();
        // periodic Hann of 16 sums to 8
        let mid = &spec.data[4 * spec.bins];
        assert!((mid.re - 8.0).abs() < 1e-5 && mid.im.abs() < 1e-5);
    }
}
