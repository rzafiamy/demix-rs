//! Music source separation (BS-RoFormer) in Rust/Candle, weights in one GGUF.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let device = candle_core::Device::Cpu;
//! let sep = demix::Separator::open("models/bs-roformer-viperx-317-f16.gguf".as_ref(), &device, None)?;
//! let audio = demix::audio::decode(std::fs::read("song.mp3")?, Some("mp3"))?;
//! let mix = demix::audio::conform(audio, sep.cfg.channels(), sep.cfg.sample_rate)?;
//! for stem in sep.separate(&mix, |_, _| {})? {
//!     let wav = demix::audio::encode_wav(&stem.audio, sep.cfg.sample_rate, demix::audio::SampleFormat::S16)?;
//!     std::fs::write(format!("{}.wav", stem.name), wav)?;
//! }
//! # Ok(()) }
//! ```

pub mod audio;
pub mod config;
pub mod gguf;
pub mod model;
pub mod separate;
pub mod stft;
pub mod weights;

pub use config::Config;
pub use separate::{Separator, Stem};

use anyhow::Result;
use candle_core::{DType, Device};
use std::path::Path;

impl Separator {
    /// Opens a demix GGUF. `dtype` defaults to F16 on a GPU, F32 on the CPU.
    pub fn open(path: &Path, device: &Device, dtype: Option<DType>) -> Result<Self> {
        let (w, cfg, _) = gguf::load(path, device)?;
        let dtype = dtype.unwrap_or(if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        });
        let model = model::BsRoformer::load(&w, &cfg, dtype)?;
        Ok(Self::new(cfg, model))
    }
}

/// The device for `cpu` = false: CUDA (or Metal) when built in, else the CPU.
pub fn pick_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(Device::new_metal(0)?);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}
