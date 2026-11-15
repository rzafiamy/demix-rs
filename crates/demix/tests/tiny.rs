//! End to end on a tiny random BS-RoFormer (no model download): checkpoint
//! (safetensors) -> GGUF -> separation; the F32 GGUF must give exactly what
//! the checkpoint gives, Q8_0 / F16 must stay close.

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use demix::Separator;
use demix::separate::Planar;
use std::collections::HashMap;
use std::path::Path;

const YAML: &str = "audio:
  chunk_size: 400
  sample_rate: 8000
model:
  dim: 32
  depth: 2
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  freqs_per_bands: !!python/tuple
    - 4
    - 5
  dim_head: 16
  heads: 2
  stft_n_fft: 16
  stft_hop_length: 4
  stft_win_length: 16
  mask_estimator_depth: 2
training:
  instruments: [vocals, other]
  target_instrument: vocals
inference:
  num_overlap: 2
";

fn rand(shape: &[usize], scale: f64, seed: &mut u64) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|_| {
            // xorshift: deterministic, no rand dependency
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed % 20_000) as f64 / 10_000.0 - 1.0) as f32 * scale as f32
        })
        .collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn write_checkpoint(path: &Path) {
    let (dim, h, dh, hidden) = (32usize, 2usize, 16usize, 128usize);
    let bands = [4usize * 2 * 2, 5 * 2 * 2];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let ones = |n: usize| Tensor::ones(n, DType::F32, &Device::Cpu).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let p = format!("layers.{i}.{j}.layers.0");
            let freqs: Vec<f32> = (0..dh / 2)
                .map(|k| 1.0 / 10_000f32.powf(2.0 * k as f32 / dh as f32))
                .collect();
            t.insert(
                format!("{p}.0.rotary_embed.freqs"),
                Tensor::new(freqs, &Device::Cpu).unwrap(),
            );
            t.insert(format!("{p}.0.norm.gamma"), ones(dim));
            t.insert(
                format!("{p}.0.to_qkv.weight"),
                rand(&[3 * h * dh, dim], 0.2, &mut s),
            );
            t.insert(
                format!("{p}.0.to_gates.weight"),
                rand(&[h, dim], 0.2, &mut s),
            );
            t.insert(format!("{p}.0.to_gates.bias"), rand(&[h], 0.2, &mut s));
            t.insert(
                format!("{p}.0.to_out.0.weight"),
                rand(&[dim, h * dh], 0.2, &mut s),
            );
            t.insert(format!("{p}.1.net.0.gamma"), ones(dim));
            t.insert(
                format!("{p}.1.net.1.weight"),
                rand(&[4 * dim, dim], 0.2, &mut s),
            );
            t.insert(format!("{p}.1.net.1.bias"), rand(&[4 * dim], 0.1, &mut s));
            t.insert(
                format!("{p}.1.net.4.weight"),
                rand(&[dim, 4 * dim], 0.1, &mut s),
            );
            t.insert(format!("{p}.1.net.4.bias"), rand(&[dim], 0.1, &mut s));
        }
    }
    t.insert("final_norm.gamma".into(), ones(dim));
    for (b, &d) in bands.iter().enumerate() {
        let p = format!("band_split.to_features.{b}");
        t.insert(format!("{p}.0.gamma"), ones(d));
        t.insert(format!("{p}.1.weight"), rand(&[dim, d], 0.3, &mut s));
        t.insert(format!("{p}.1.bias"), rand(&[dim], 0.1, &mut s));
        let p = format!("mask_estimators.0.to_freqs.{b}.0");
        t.insert(format!("{p}.0.weight"), rand(&[hidden, dim], 0.2, &mut s));
        t.insert(format!("{p}.0.bias"), rand(&[hidden], 0.1, &mut s));
        t.insert(format!("{p}.2.weight"), rand(&[2 * d, hidden], 0.1, &mut s));
        t.insert(format!("{p}.2.bias"), rand(&[2 * d], 0.1, &mut s));
    }
    candle_core::safetensors::save(&t, path).unwrap();
}

fn mix(len: usize) -> Planar {
    (0..2)
        .map(|c| {
            (0..len)
                .map(|i| {
                    ((i as f32) * (0.05 + 0.01 * c as f32)).sin() * 0.4
                        + ((i * 31 % 17) as f32 / 17.0 - 0.5) * 0.1
                })
                .collect()
        })
        .collect()
}

fn max_diff(a: &Planar, b: &Planar) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn checkpoint_gguf_and_separation_agree() {
    let dir = tempdir();
    let ckpt = dir.join("tiny.safetensors");
    let yaml = dir.join("tiny.yaml");
    write_checkpoint(&ckpt);
    std::fs::write(&yaml, YAML).unwrap();

    let (w, cfg) = demix::gguf::load_checkpoint(&ckpt, &yaml, &Device::Cpu).unwrap();
    let reference = Separator::new(
        cfg.clone(),
        demix::model::BsRoformer::load(&w, &cfg, DType::F32).unwrap(),
    );
    assert_eq!(reference.stem_names(), vec!["vocals", "instrumental"]);

    // long enough for the reflect-padded path, plus a short one (no padding,
    // last chunk zero-padded)
    for len in [1_500usize, 300] {
        let m = mix(len);
        let want = reference.separate(&m, |_, _| {}).unwrap();
        assert_eq!(want.len(), 2);
        assert_eq!(want[0].audio.len(), 2);
        assert_eq!(want[0].audio[0].len(), len);
        assert!(want[0].audio.iter().flatten().all(|v| v.is_finite()));
        // instrumental = mix - vocals
        let inst: Planar = m
            .iter()
            .zip(&want[0].audio)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x - y).collect())
            .collect();
        assert!(max_diff(&inst, &want[1].audio) < 1e-6);

        for (dt, tol) in [
            (GgmlDType::F32, 0.0f32),
            (GgmlDType::F16, 2e-2),
            (GgmlDType::Q8_0, 5e-2),
        ] {
            let out = dir.join(format!("tiny-{dt:?}.gguf"));
            demix::gguf::convert(&ckpt, &yaml, &out, dt, "tiny").unwrap();
            let sep = Separator::open(&out, &Device::Cpu, Some(DType::F32)).unwrap();
            assert_eq!(sep.cfg, cfg);
            let got = sep.separate(&m, |_, _| {}).unwrap();
            let d = max_diff(&want[0].audio, &got[0].audio);
            assert!(d <= tol, "{dt:?} len {len}: max diff {d}");
        }
    }

    // batching chunks does not change the result
    let m = mix(1_500);
    let mut batched = Separator::new(
        cfg.clone(),
        demix::model::BsRoformer::load(&w, &cfg, DType::F32).unwrap(),
    );
    batched.batch_size = 3;
    let a = reference.separate(&m, |_, _| {}).unwrap();
    let b = batched.separate(&m, |_, _| {}).unwrap();
    assert!(max_diff(&a[0].audio, &b[0].audio) < 1e-5);

    // slicing the transformers under a tiny memory budget does not either
    let mut sliced = Separator::new(
        cfg.clone(),
        demix::model::BsRoformer::load(&w, &cfg, DType::F32).unwrap(),
    );
    sliced.model.attn_budget = 1;
    let c = sliced.separate(&m, |_, _| {}).unwrap();
    assert!(max_diff(&a[0].audio, &c[0].audio) < 1e-5);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wrong_config_is_refused_at_conversion() {
    let dir = tempdir();
    let ckpt = dir.join("tiny.safetensors");
    let yaml = dir.join("bad.yaml");
    write_checkpoint(&ckpt);
    std::fs::write(&yaml, YAML.replace("  depth: 2", "  depth: 3")).unwrap();
    let err =
        demix::gguf::convert(&ckpt, &yaml, &dir.join("x.gguf"), GgmlDType::F16, "x").unwrap_err();
    assert!(format!("{err:#}").contains("does not match"), "{err:#}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn tempdir() -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "demix-test-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}
