//! Audio in and out, pure Rust: decoding (WAV, FLAC, MP3, Ogg Vorbis,
//! AAC/ALAC in MP4/M4A) with symphonia, resampling with rubato, WAV writing
//! with hound.

use crate::separate::Planar;
use anyhow::{Context, Result, bail};
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct Audio {
    pub sample_rate: u32,
    /// `[channel][sample]`.
    pub data: Planar,
}

impl Audio {
    pub fn duration_secs(&self) -> f64 {
        self.data
            .first()
            .map_or(0.0, |c| c.len() as f64 / self.sample_rate as f64)
    }
}

/// Decodes an in-memory audio file. `ext` (e.g. "mp3") helps the probe.
pub fn decode(bytes: Vec<u8>, ext: Option<&str>) -> Result<Audio> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    if let Some(e) = ext {
        hint.with_extension(e);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unsupported or unreadable audio (wav, flac, mp3, ogg, m4a)")?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut data: Planar = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e).context("reading audio"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // a corrupt frame: skip it, as ffmpeg does
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding audio"),
        };
        let spec = *decoded.spec();
        sample_rate = spec.rate;
        let ch = spec.channels.count();
        if data.is_empty() {
            data = vec![Vec::new(); ch];
        }
        let b = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if b.capacity() < decoded.capacity() * ch {
            *b = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        b.copy_interleaved_ref(decoded);
        for frame in b.samples().chunks_exact(ch) {
            for (c, v) in frame.iter().enumerate() {
                data[c].push(*v);
            }
        }
    }
    if data.is_empty() || data[0].is_empty() || sample_rate == 0 {
        bail!("no audio decoded");
    }
    Ok(Audio { sample_rate, data })
}

/// Converts to `channels` channels (mono is duplicated, as MSST does; extra
/// channels are dropped) at `rate` Hz.
pub fn conform(audio: Audio, channels: usize, rate: u32) -> Result<Planar> {
    let mut data = audio.data;
    if channels == 1 && data.len() > 1 {
        let n = data[0].len();
        let k = data.len() as f32;
        let mono = (0..n)
            .map(|i| data.iter().map(|c| c[i]).sum::<f32>() / k)
            .collect();
        data = vec![mono];
    } else if data.len() == 1 && channels == 2 {
        data.push(data[0].clone());
    } else {
        data.truncate(channels);
    }
    if audio.sample_rate != rate {
        data = resample(&data, audio.sample_rate, rate)?;
    }
    Ok(data)
}

/// Band-limited sinc resampling of planar audio.
pub fn resample(data: &Planar, from: u32, to: u32) -> Result<Planar> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 8192;
    let ratio = to as f64 / from as f64;
    let mut rs = SincFixedIn::<f32>::new(ratio, 1.0, params, chunk, data.len())?;
    let n = data[0].len();
    let delay = rs.output_delay();
    let want = (n as f64 * ratio).round() as usize;
    let mut out: Planar = vec![Vec::with_capacity(want + chunk); data.len()];
    let mut pos = 0;
    while pos < n {
        let end = (pos + chunk).min(n);
        let block: Vec<&[f32]> = data.iter().map(|c| &c[pos..end]).collect();
        let res = if end - pos == chunk {
            rs.process(&block, None)?
        } else {
            rs.process_partial(Some(&block), None)?
        };
        for (o, r) in out.iter_mut().zip(res) {
            o.extend_from_slice(&r);
        }
        pos = end;
    }
    // flush the filter delay
    while out[0].len() < want + delay {
        let res = rs.process_partial::<&[f32]>(None, None)?;
        if res[0].is_empty() {
            break;
        }
        for (o, r) in out.iter_mut().zip(res) {
            o.extend_from_slice(&r);
        }
    }
    for o in out.iter_mut() {
        let end = (delay + want).min(o.len());
        *o = o[delay.min(end)..end].to_vec();
        o.resize(want, 0.0);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit PCM (clipped).
    S16,
    /// 32-bit float.
    F32,
}

impl std::str::FromStr for SampleFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "s16" | "pcm16" | "16" | "wav" => Self::S16,
            "f32" | "float" | "32" => Self::F32,
            o => bail!("unknown sample format '{o}' (s16, f32)"),
        })
    }
}

/// Encodes planar audio as a WAV file in memory.
pub fn encode_wav(data: &Planar, rate: u32, fmt: SampleFormat) -> Result<Vec<u8>> {
    let ch = data.len();
    let spec = hound::WavSpec {
        channels: ch as u16,
        sample_rate: rate,
        bits_per_sample: if fmt == SampleFormat::S16 { 16 } else { 32 },
        sample_format: if fmt == SampleFormat::S16 {
            hound::SampleFormat::Int
        } else {
            hound::SampleFormat::Float
        },
    };
    let n = data.first().map_or(0, Vec::len);
    let mut cur = Cursor::new(Vec::with_capacity(n * ch * 4 + 64));
    {
        let mut w = hound::WavWriter::new(&mut cur, spec)?;
        for i in 0..n {
            for c in data {
                match fmt {
                    SampleFormat::S16 => {
                        let v = (c[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
                        w.write_sample(v)?;
                    }
                    SampleFormat::F32 => w.write_sample(c[i])?,
                }
            }
        }
        w.finalize()?;
    }
    Ok(cur.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_and_conform() {
        let sr = 22_050;
        let tone: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let bytes = encode_wav(&vec![tone.clone()], sr as u32, SampleFormat::F32).unwrap();
        let a = decode(bytes, Some("wav")).unwrap();
        assert_eq!(a.sample_rate, sr as u32);
        assert_eq!(a.data.len(), 1);
        assert_eq!(a.data[0].len(), tone.len());
        let st = conform(a, 2, 44_100).unwrap();
        assert_eq!(st.len(), 2);
        assert_eq!(st[0].len(), 2 * tone.len());
        // the resampled tone keeps its amplitude away from the edges
        let peak = st[0][1000..40_000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
    }
}
