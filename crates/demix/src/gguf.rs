//! Single-file GGUF models, and loading straight from a PyTorch checkpoint.
//!
//! GGUF layout:
//! - metadata `general.architecture = "demix"`, `demix.config` (the
//!   [`Config`] as JSON), `general.name`;
//! - every checkpoint tensor under its original name, minus the duplicated
//!   rotary frequencies (one copy per transformer is enough; they are kept as
//!   `layers.N.M.layers.0.0.rotary_embed.freqs`). 2-D linear weights whose
//!   rows fit the block use the requested dtype (else Q8_0, F16 or F32);
//!   norms, biases and rotary frequencies stay F32.
//!
//! Checkpoints are read with candle's pickle reader (tensors only: no code
//! from the file runs), either a bare state dict or one under `state_dict`.

use crate::config::Config;
use crate::weights::Weights;
use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub const ARCH: &str = "demix";

/// Parses a dtype name: f32, f16, bf16, q8_0.
pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        other => bail!("unknown dtype '{other}' (f32, f16, bf16, q8_0)"),
    })
}

fn fit_dtype(wanted: GgmlDType, row: usize) -> GgmlDType {
    for dt in [wanted, GgmlDType::Q8_0, GgmlDType::F16] {
        if row.is_multiple_of(dt.block_size()) {
            return dt;
        }
    }
    GgmlDType::F32
}

/// Tensors of a PyTorch checkpoint (`.ckpt` / `.pt` / `.pth`) or a
/// `.safetensors` file, as F32 on the CPU.
pub fn read_checkpoint(path: &Path) -> Result<Vec<(String, Tensor)>> {
    let tensors = if path.extension().is_some_and(|e| e == "safetensors") {
        candle_core::safetensors::load(path, &Device::Cpu)?
            .into_iter()
            .collect()
    } else {
        let mut t = candle_core::pickle::read_all(path)
            .with_context(|| format!("reading checkpoint {path:?}"))?;
        if !t.iter().any(|(n, _)| n.starts_with("band_split.")) {
            t = candle_core::pickle::read_all_with_key(path, Some("state_dict"))
                .with_context(|| format!("reading {path:?} (state_dict)"))?;
        }
        t
    };
    if !tensors.iter().any(|(n, _)| n.starts_with("band_split.")) {
        bail!("{path:?} holds no BS-RoFormer weights (no band_split.*)");
    }
    tensors
        .into_iter()
        .map(|(n, t)| {
            // torch.compile prefix, as MSST's strip_compile_prefix
            let n = n.strip_prefix("_orig_mod.").map(String::from).unwrap_or(n);
            Ok((n, t.to_dtype(DType::F32)?))
        })
        .collect()
}

/// Rotary frequencies other than each transformer's first copy.
fn redundant(name: &str) -> bool {
    name.ends_with("rotary_embed.freqs") && !name.contains(".layers.0.0.rotary_embed")
}

/// Converts a checkpoint + MSST YAML config to one GGUF; returns per-dtype
/// tensor counts.
pub fn convert(
    ckpt: &Path,
    yaml: &Path,
    out: &Path,
    dtype: GgmlDType,
    name: &str,
) -> Result<Vec<(String, usize)>> {
    let text = std::fs::read_to_string(yaml).with_context(|| format!("reading {yaml:?}"))?;
    let cfg = Config::from_msst_yaml(&text)?;
    let mut tensors = read_checkpoint(ckpt)?;
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    // Build the model once on the CPU: fails here, not at load, when the
    // checkpoint does not match the config.
    {
        let map = to_map(&tensors)?;
        let w = Weights::new(map, Device::Cpu);
        crate::model::BsRoformer::load(&w, &cfg, DType::F32)
            .context("checkpoint does not match the config")?;
    }

    let mut q = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (n, t) in &tensors {
        if redundant(n) {
            continue;
        }
        let dt = if t.rank() == 2 && n.ends_with(".weight") {
            fit_dtype(dtype, t.dim(1)?)
        } else {
            GgmlDType::F32
        };
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        q.push((n.clone(), QTensor::quantize(t, dt)?));
    }
    let meta = [
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        ("general.name", gguf_file::Value::String(name.into())),
        (
            "demix.config",
            gguf_file::Value::String(serde_json::to_string(&cfg)?),
        ),
    ];
    let meta: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = q.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &meta, &refs)?;
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(counts)
}

fn to_map(tensors: &[(String, Tensor)]) -> Result<HashMap<String, Arc<QTensor>>> {
    tensors
        .iter()
        .map(|(n, t)| Ok((n.clone(), Arc::new(QTensor::quantize(t, GgmlDType::F32)?))))
        .collect()
}

/// Loads a checkpoint with its MSST YAML config (F32, for parity checks).
pub fn load_checkpoint(ckpt: &Path, yaml: &Path, device: &Device) -> Result<(Weights, Config)> {
    let text = std::fs::read_to_string(yaml).with_context(|| format!("reading {yaml:?}"))?;
    let cfg = Config::from_msst_yaml(&text)?;
    let map = to_map(&read_checkpoint(ckpt)?)?;
    Ok((Weights::new(map, device.clone()), cfg))
}

/// Loads a demix GGUF.
pub fn load(path: &Path, device: &Device) -> Result<(Weights, Config, String)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let content =
        gguf_file::Content::read(&mut file).with_context(|| format!("reading {path:?}"))?;
    let meta = |k: &str| -> Result<String> {
        match content.metadata.get(k) {
            Some(gguf_file::Value::String(s)) => Ok(s.clone()),
            _ => bail!("{path:?}: missing metadata '{k}'"),
        }
    };
    if meta("general.architecture")? != ARCH {
        bail!("{path:?} is not a {ARCH} GGUF");
    }
    let cfg = Config::from_json(&meta("demix.config")?)?;
    let name = meta("general.name").unwrap_or_default();
    let mut map = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for n in names {
        // quantized tensors go straight to the device (used as is); dense
        // ones stay on the CPU until a layer converts them
        let quantized = !matches!(
            content.tensor_infos[&n].ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        let dev = if quantized { device } else { &Device::Cpu };
        map.insert(n.clone(), Arc::new(content.tensor(&mut file, &n, dev)?));
    }
    Ok((Weights::new(map, device.clone()), cfg, name))
}
