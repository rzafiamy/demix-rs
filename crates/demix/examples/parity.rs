//! Compares the Rust port with a dump of `scripts/ref/ref.py`.
//!
//! usage: parity <model.gguf | model.ckpt config.yaml> <ref.safetensors> [--cpu] [--dtype f32|f16|bf16]
//!
//! Prints, per tensor, the SNR in dB (10 log10 |ref|^2 / |ref - rust|^2) and
//! the max abs difference: intermediates of the first chunk, the chunk's
//! output, and the whole-track `sources`.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use demix::Separator;
use demix::separate::Planar;
use std::path::PathBuf;

fn snr(r: &[f32], x: &[f32]) -> (f64, f64) {
    let (mut s, mut n, mut m) = (0f64, 0f64, 0f64);
    for (a, b) in r.iter().zip(x) {
        let (a, b) = (*a as f64, *b as f64);
        s += a * a;
        n += (a - b) * (a - b);
        m = m.max((a - b).abs());
    }
    (10.0 * (s / n.max(1e-30)).log10(), m)
}

fn report(name: &str, r: &Tensor, x: &Tensor) -> Result<f64> {
    let r = r.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let x = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    if r.elem_count() != x.elem_count() {
        bail!(
            "{name}: shapes differ, ref {:?} vs rust {:?}",
            r.dims(),
            x.dims()
        );
    }
    let (db, m) = snr(&r.flatten_all()?.to_vec1()?, &x.flatten_all()?.to_vec1()?);
    println!(
        "{name:<12} {:>20}  SNR {db:7.2} dB  max|diff| {m:.3e}",
        format!("{:?}", r.dims())
    );
    Ok(db)
}

fn planar(t: &Tensor) -> Result<Planar> {
    let (c, _) = t.dims2()?;
    (0..c).map(|i| Ok(t.get(i)?.to_vec1::<f32>()?)).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let dtype = match args
        .iter()
        .position(|a| a == "--dtype")
        .map(|i| args[i + 1].as_str())
    {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let pos: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && !(i > &0 && args[i - 1] == "--dtype"))
        .map(|(_, a)| a)
        .collect();
    let device = demix::pick_device(cpu)?;
    let (w, cfg, refp) = match pos.as_slice() {
        [m, r] => {
            let (w, c, _) = demix::gguf::load(&PathBuf::from(m), &device)?;
            (w, c, PathBuf::from(r))
        }
        [ck, y, r] => {
            let (w, c) =
                demix::gguf::load_checkpoint(&PathBuf::from(ck), &PathBuf::from(y), &device)?;
            (w, c, PathBuf::from(r))
        }
        _ => bail!(
            "usage: parity <model.gguf | model.ckpt config.yaml> <ref.safetensors> [--cpu] [--dtype f32|f16|bf16]"
        ),
    };
    let model = demix::model::BsRoformer::load(&w, &cfg, dtype)?;
    let sep = Separator::new(cfg, model);
    let r = candle_core::safetensors::load(&refp, &Device::Cpu).context("reading the reference")?;
    println!("device {device:?}, dtype {dtype:?}");

    let mut worst = f64::INFINITY;
    if let Some(chunk_in) = r.get("chunk_in") {
        // model intermediates on the first chunk
        let ch_planar = planar(chunk_in)?;
        let stft = demix::stft::Stft::new(
            sep.cfg.stft_n_fft,
            sep.cfg.stft_hop_length,
            sep.cfg.stft_normalized,
        );
        let ch = sep.cfg.channels();
        let specs: Vec<_> = ch_planar
            .iter()
            .map(|c| stft.forward(c))
            .collect::<Result<_>>()?;
        let (bins, frames) = (specs[0].bins, specs[0].frames);
        let width = bins * ch * 2;
        let mut feat = vec![0f32; frames * width];
        for t in 0..frames {
            for f in 0..bins {
                for (s, sp) in specs.iter().enumerate() {
                    let z = sp.data[t * bins + f];
                    feat[t * width + (f * ch + s) * 2] = z.re;
                    feat[t * width + (f * ch + s) * 2 + 1] = z.im;
                }
            }
        }
        let x = Tensor::from_vec(feat, (1, frames, width), &device)?;
        for (name, t) in sep.model.forward_debug(&x)? {
            if let Some(rt) = r.get(name) {
                worst = worst.min(report(name, rt, &t)?);
            }
        }
        let out = sep.run_chunks(&[ch_planar])?;
        let flat: Vec<f32> = out[0].iter().flatten().flatten().copied().collect();
        let n = flat.len();
        let t = Tensor::from_vec(flat, n, &Device::Cpu)?;
        worst = worst.min(report("chunk_out", &r["chunk_out"], &t)?);
    }
    let mix = planar(&r["mix"])?;
    let t0 = std::time::Instant::now();
    let stems = sep.separate(&mix, |_, _| {})?;
    let el = t0.elapsed().as_secs_f64();
    let nstem = sep.cfg.num_stems;
    let flat: Vec<f32> = stems[..nstem]
        .iter()
        .flat_map(|s| s.audio.iter().flatten().copied())
        .collect();
    let n = flat.len();
    let t = Tensor::from_vec(flat, n, &Device::Cpu)?;
    worst = worst.min(report("sources", &r["sources"], &t)?);
    println!("whole track in {el:.2}s; worst SNR {worst:.2} dB");
    Ok(())
}
