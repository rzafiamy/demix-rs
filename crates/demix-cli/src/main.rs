//! `demix`: convert a BS-RoFormer checkpoint to GGUF, separate files, or
//! serve the HTTP API.

mod server;

use anyhow::{Context, Result, bail};
use candle_core::DType;
use clap::{Parser, Subcommand};
use demix::Separator;
use demix::audio::{self, SampleFormat};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "demix",
    version,
    about = "Music source separation (BS-RoFormer), GGUF weights"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct RunOpts {
    /// demix GGUF (from `demix convert`)
    #[arg(short, long, env = "DEMIX_MODEL")]
    model: PathBuf,
    /// Run on the CPU even when a GPU build finds a GPU
    #[arg(long)]
    cpu: bool,
    /// Compute dtype: f32, f16, bf16 (default: f16 on a GPU, f32 on the CPU)
    #[arg(long)]
    dtype: Option<String>,
    /// Chunks per model call (more is faster on a GPU, needs more memory)
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Overlapping passes per sample (config default: usually 2; 4 is
    /// slightly cleaner and twice as slow)
    #[arg(long)]
    overlap: Option<usize>,
    /// CPU threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    /// MiB the attention scores of one call may take (lower: less GPU
    /// memory, a little slower)
    #[arg(long, default_value_t = 1024)]
    attn_budget_mb: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a checkpoint (.ckpt/.pth/.safetensors) and its MSST YAML config to one GGUF
    Convert {
        checkpoint: PathBuf,
        config: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Linear weight dtype: f16 (default), q8_0, bf16, f32
        #[arg(long, default_value = "f16")]
        dtype: String,
        /// Model name stored in the file (default: output file stem)
        #[arg(long)]
        name: Option<String>,
    },
    /// Separate audio files into stems (WAV, at the model's sample rate)
    Separate {
        #[command(flatten)]
        run: RunOpts,
        /// Input files (wav, flac, mp3, ogg, m4a)
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output directory (files: <name>_<stem>.wav)
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
        /// Stems to write, comma-separated (default: all)
        #[arg(long, value_delimiter = ',')]
        stems: Vec<String>,
        /// Sample format: s16 (default) or f32
        #[arg(long, default_value = "s16")]
        format: SampleFormat,
    },
    /// Serve the HTTP API (POST /v1/audio/separations)
    Serve {
        #[command(flatten)]
        run: RunOpts,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8790)]
        port: u16,
        /// Model id reported by /v1/models (default: the GGUF's name)
        #[arg(long)]
        model_id: Option<String>,
    },
}

fn parse_compute_dtype(s: &str) -> Result<DType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        o => bail!("unknown compute dtype '{o}' (f32, f16, bf16)"),
    })
}

fn open(run: &RunOpts) -> Result<(Separator, String)> {
    if let Some(n) = run.threads {
        // candle's CPU kernels use rayon's global pool, which reads this
        // once, at its first use (later than here)
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
    let device = demix::pick_device(run.cpu)?;
    let dtype = run.dtype.as_deref().map(parse_compute_dtype).transpose()?;
    let t = Instant::now();
    let (w, mut cfg, name) = demix::gguf::load(&run.model, &device)?;
    if let Some(o) = run.overlap {
        if o == 0 {
            bail!("--overlap must be >= 1");
        }
        cfg.num_overlap = o;
    }
    let dtype = dtype.unwrap_or(if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    });
    let mut model = demix::model::BsRoformer::load(&w, &cfg, dtype)?;
    model.attn_budget = run.attn_budget_mb.max(16) << 20;
    let mut sep = Separator::new(cfg, model);
    sep.batch_size = run.batch.max(1);
    let name = if name.is_empty() {
        run.model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "demix".into())
    } else {
        name
    };
    eprintln!(
        "loaded {name} ({}, {dtype:?}) in {:.1}s; stems: {}",
        device_name(&device),
        t.elapsed().as_secs_f64(),
        sep.stem_names().join(", ")
    );
    Ok((sep, name))
}

fn device_name(d: &candle_core::Device) -> &'static str {
    if d.is_cuda() {
        "cuda"
    } else if d.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

fn ext_of(p: &Path) -> Option<String> {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
}

fn separate_files(
    run: &RunOpts,
    inputs: &[PathBuf],
    out: &Path,
    stems: &[String],
    fmt: SampleFormat,
) -> Result<()> {
    let (sep, _) = open(run)?;
    let names = sep.stem_names();
    for s in stems {
        if !names.contains(s) {
            bail!("unknown stem '{s}' (this model: {})", names.join(", "));
        }
    }
    std::fs::create_dir_all(out)?;
    for input in inputs {
        let t = Instant::now();
        let bytes = std::fs::read(input).with_context(|| format!("reading {input:?}"))?;
        let a = audio::decode(bytes, ext_of(input).as_deref())
            .with_context(|| format!("decoding {input:?}"))?;
        let secs = a.duration_secs();
        let mix = audio::conform(a, sep.cfg.channels(), sep.cfg.sample_rate)?;
        let result = sep.separate(&mix, |d, n| eprint!("\r{}: chunk {d}/{n}", input.display()))?;
        eprintln!();
        let base = input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".into());
        for stem in result {
            if !stems.is_empty() && !stems.contains(&stem.name) {
                continue;
            }
            let path = out.join(format!("{base}_{}.wav", stem.name));
            std::fs::write(
                &path,
                audio::encode_wav(&stem.audio, sep.cfg.sample_rate, fmt)?,
            )?;
            eprintln!("  {}", path.display());
        }
        let el = t.elapsed().as_secs_f64();
        eprintln!(
            "{}: {secs:.1}s of audio in {el:.1}s ({:.1}x realtime)",
            input.display(),
            secs / el
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,symphonia=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Convert {
            checkpoint,
            config,
            output,
            dtype,
            name,
        } => {
            let dt = demix::gguf::parse_dtype(&dtype)?;
            let name = name.unwrap_or_else(|| {
                output
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            let t = Instant::now();
            let counts = demix::gguf::convert(&checkpoint, &config, &output, dt, &name)?;
            let size = std::fs::metadata(&output)?.len() as f64 / 1e6;
            eprintln!(
                "wrote {} ({size:.0} MB) in {:.1}s: {}",
                output.display(),
                t.elapsed().as_secs_f64(),
                counts
                    .iter()
                    .map(|(d, n)| format!("{n} {d}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Cmd::Separate {
            run,
            inputs,
            output,
            stems,
            format,
        } => separate_files(&run, &inputs, &output, &stems, format)?,
        Cmd::Serve {
            run,
            host,
            port,
            model_id,
        } => {
            let (sep, name) = open(&run)?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(server::serve(sep, model_id.unwrap_or(name), &host, port))?;
        }
    }
    Ok(())
}
