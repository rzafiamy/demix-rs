//! Model and inference settings, read from a Music-Source-Separation-Training
//! YAML config (`model:`, `audio:`, `training:`, `inference:`) and stored as
//! JSON in the GGUF (`demix.config`).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use yaml_rust2::{Yaml, YamlLoader};

pub const ARCH_BS_ROFORMER: &str = "bs_roformer";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// `bs_roformer`.
    pub arch: String,
    pub dim: usize,
    pub depth: usize,
    pub stereo: bool,
    pub num_stems: usize,
    pub time_transformer_depth: usize,
    pub freq_transformer_depth: usize,
    pub linear_transformer_depth: usize,
    pub freqs_per_bands: Vec<usize>,
    pub dim_head: usize,
    pub heads: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    pub stft_normalized: bool,
    pub zero_dc: bool,
    pub mask_estimator_depth: usize,
    pub mlp_expansion_factor: usize,
    pub skip_connection: bool,
    /// Rotary base when the checkpoint holds no `rotary_embed.freqs`.
    pub rope_theta: f64,

    pub sample_rate: u32,
    pub chunk_size: usize,
    pub num_overlap: usize,
    /// Names of the model's outputs, in order (`num_stems` of them).
    pub stems: Vec<String>,
}

impl Config {
    pub fn channels(&self) -> usize {
        if self.stereo { 2 } else { 1 }
    }

    pub fn freq_bins(&self) -> usize {
        self.stft_n_fft / 2 + 1
    }

    /// Input width of each band: freqs x channels x (re, im).
    pub fn band_dims(&self) -> Vec<usize> {
        self.freqs_per_bands
            .iter()
            .map(|f| 2 * f * self.channels())
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.arch != ARCH_BS_ROFORMER {
            bail!("unsupported architecture '{}' (bs_roformer)", self.arch);
        }
        if self.linear_transformer_depth != 0 {
            bail!("linear_transformer_depth > 0 is not supported yet");
        }
        let sum: usize = self.freqs_per_bands.iter().sum();
        if sum != self.freq_bins() {
            bail!(
                "freqs_per_bands sum to {sum}, the STFT gives {} bins",
                self.freq_bins()
            );
        }
        if self.stft_win_length != self.stft_n_fft {
            bail!("stft_win_length != stft_n_fft is not supported");
        }
        if self.stems.len() != self.num_stems {
            bail!(
                "{} stem names for num_stems = {}",
                self.stems.len(),
                self.num_stems
            );
        }
        if self.num_overlap == 0 || self.chunk_size < self.num_overlap * 10 {
            bail!("bad chunk_size / num_overlap");
        }
        if !self.dim_head.is_multiple_of(2) || self.heads == 0 {
            bail!("bad heads / dim_head");
        }
        Ok(())
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let c: Config = serde_json::from_str(s).context("parsing demix.config")?;
        c.validate()?;
        Ok(c)
    }

    /// Reads an MSST YAML config (`!!python/tuple` tags are plain lists).
    pub fn from_msst_yaml(text: &str) -> Result<Self> {
        let docs = YamlLoader::load_from_str(text).context("parsing YAML")?;
        let root = yaml_to_json(docs.first().context("empty YAML")?);
        let section = |k: &str| {
            root.get(k)
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default()
        };
        let model = section("model");
        let audio = section("audio");
        let training = section("training");
        let inference = section("inference");
        if model.is_empty() {
            bail!("no `model:` section");
        }
        if model.contains_key("num_bands") {
            bail!("this is a Mel-Band RoFormer config: not supported yet");
        }
        if model.get("use_pope").and_then(Value::as_bool) == Some(true) {
            bail!("use_pope is not supported");
        }

        let u = |m: &Map<String, Value>, k: &str, d: usize| -> Result<usize> {
            match m.get(k) {
                None | Some(Value::Null) => Ok(d),
                Some(v) => v
                    .as_u64()
                    .map(|n| n as usize)
                    .with_context(|| format!("`{k}` must be an integer")),
            }
        };
        let b = |m: &Map<String, Value>, k: &str, d: bool| -> bool {
            m.get(k).and_then(Value::as_bool).unwrap_or(d)
        };

        let freqs_per_bands = match model.get("freqs_per_bands") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .context("freqs_per_bands must be integers")?,
            _ => DEFAULT_FREQS_PER_BANDS.to_vec(),
        };
        let num_stems = u(&model, "num_stems", 1)?;

        // MSST `prefer_target_instrument`: the target instrument alone when
        // set, else every instrument.
        let instruments: Vec<String> = training
            .get("instruments")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let stems = match training.get("target_instrument").and_then(Value::as_str) {
            Some(t) if !t.is_empty() => vec![t.to_string()],
            _ => instruments,
        };
        let stems = if stems.len() == num_stems {
            stems
        } else {
            (0..num_stems).map(|i| format!("stem{i}")).collect()
        };

        let chunk_size = match inference.get("chunk_size") {
            Some(v) if v.is_u64() => v.as_u64().unwrap() as usize,
            _ => u(&audio, "chunk_size", 352_800)?,
        };

        let c = Config {
            arch: ARCH_BS_ROFORMER.into(),
            dim: u(&model, "dim", 0)?,
            depth: u(&model, "depth", 0)?,
            stereo: b(&model, "stereo", false),
            num_stems,
            time_transformer_depth: u(&model, "time_transformer_depth", 2)?,
            freq_transformer_depth: u(&model, "freq_transformer_depth", 2)?,
            linear_transformer_depth: u(&model, "linear_transformer_depth", 0)?,
            freqs_per_bands,
            dim_head: u(&model, "dim_head", 64)?,
            heads: u(&model, "heads", 8)?,
            stft_n_fft: u(&model, "stft_n_fft", 2048)?,
            stft_hop_length: u(&model, "stft_hop_length", 512)?,
            stft_win_length: u(&model, "stft_win_length", 2048)?,
            stft_normalized: b(&model, "stft_normalized", false),
            zero_dc: b(&model, "zero_dc", true),
            mask_estimator_depth: u(&model, "mask_estimator_depth", 2)?,
            mlp_expansion_factor: u(&model, "mlp_expansion_factor", 4)?,
            skip_connection: b(&model, "skip_connection", false),
            rope_theta: 10_000.0,
            sample_rate: u(&audio, "sample_rate", 44_100)? as u32,
            chunk_size,
            num_overlap: u(&inference, "num_overlap", 2)?,
            stems,
        };
        if c.dim == 0 || c.depth == 0 {
            bail!("`model.dim` and `model.depth` are required");
        }
        c.validate()?;
        Ok(c)
    }
}

/// BSRoformer's default band layout (62 bands, 1025 bins).
pub const DEFAULT_FREQS_PER_BANDS: &[usize] = &[
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];

fn yaml_to_json(y: &Yaml) -> Value {
    match y {
        Yaml::Real(s) => s
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map_or(Value::Null, Value::Number),
        Yaml::Integer(i) => Value::from(*i),
        Yaml::String(s) => Value::String(s.clone()),
        Yaml::Boolean(b) => Value::Bool(*b),
        Yaml::Array(a) => Value::Array(a.iter().map(yaml_to_json).collect()),
        Yaml::Hash(h) => Value::Object(
            h.iter()
                .filter_map(|(k, v)| {
                    let key = match k {
                        Yaml::String(s) => s.clone(),
                        Yaml::Integer(i) => i.to_string(),
                        _ => return None,
                    };
                    Some((key, yaml_to_json(v)))
                })
                .collect(),
        ),
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIPERX: &str = "audio:
  chunk_size: 352800
  sample_rate: 44100
model:
  dim: 512
  depth: 12
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
    - 2
    - 1023
  dim_head: 64
  heads: 8
  flash_attn: true
  stft_hop_length: 441
  multi_stft_normalized: False
training:
  instruments:
  - vocals
  - other
  target_instrument: vocals
  lr: 1.0e-05
inference:
  batch_size: 4
  num_overlap: 2
";

    #[test]
    fn reads_msst_yaml() {
        let c = Config::from_msst_yaml(VIPERX).unwrap();
        assert_eq!(c.dim, 512);
        assert_eq!(c.depth, 12);
        assert!(c.stereo);
        assert_eq!(c.freqs_per_bands, vec![2, 1023]);
        assert_eq!(c.stft_hop_length, 441);
        assert_eq!(c.stems, vec!["vocals"]);
        assert_eq!(c.chunk_size, 352_800);
        assert_eq!(c.num_overlap, 2);
        assert!(c.zero_dc);
        assert_eq!(c.band_dims(), vec![8, 4092]);
        let back = Config::from_json(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn rejects_mel_band_and_bad_bands() {
        let mel = VIPERX.replace("  dim: 512\n", "  dim: 512\n  num_bands: 60\n");
        assert!(Config::from_msst_yaml(&mel).is_err());
        let bad = VIPERX.replace("    - 1023\n", "    - 1000\n");
        assert!(Config::from_msst_yaml(&bad).is_err());
    }
}
