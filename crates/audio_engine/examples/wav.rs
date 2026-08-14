//! Offline speech-to-text for a WAV file.
//!
//! Feeds the whole file through the same streaming session used by the
//! `live` example.
//!
//! Usage:
//!   cargo run -p audio_engine --example wav -- --model path/to/model.gguf
//! audio.wav

use std::path::Path;

use anyhow::{Context, Result, bail};
use audio_engine::{
    Nemotron,
    dsp::to_mono_16k,
    model::{AsrModel, LatencyMode},
};

const SAMPLE_RATE: u32 = 16_000;

/// How much audio a streaming encoder step covers, trading latency against
/// throughput.
const LATENCY_MODE: LatencyMode = LatencyMode::HighQuality;

/// Read a mono WAV file as 16 kHz f32 PCM, resampling if needed.
fn read_wav_16k(path: &str) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open {path}"))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("expected mono, got {} ch", spec.channels);
    }
    let pcm: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Int => {
            let scale = match spec.bits_per_sample {
                8 => 1.0 / (i8::MAX as f32),
                16 => 1.0 / (i16::MAX as f32),
                24 | 32 => 1.0 / (i32::MAX as f32),
                b => bail!("unsupported bits_per_sample {b}"),
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!("resampling {} Hz -> 16 kHz", spec.sample_rate);
        Ok(to_mono_16k(&pcm, 1, spec.sample_rate))
    } else {
        Ok(pcm)
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    let mut wav_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next(),
            other if !other.starts_with('-') && wav_path.is_none() => {
                wav_path = Some(other.to_string());
            }
            other => bail!("unknown arg {other}"),
        }
    }
    let usage = "usage: wav --model <model.gguf> <audio.wav>";
    let model_path = model_path.with_context(|| usage)?;
    let wav_path = wav_path.with_context(|| usage)?;
    let pcm = read_wav_16k(&wav_path)?;
    println!(
        "{} samples ({:.2} s)\n",
        pcm.len(),
        pcm.len() as f64 / SAMPLE_RATE as f64
    );
    println!("Loading model...");
    let mut model = Nemotron::load(Path::new(&model_path))?;
    println!("Model loaded\n");
    let mut tr = model.live(LATENCY_MODE)?;
    let t0 = std::time::Instant::now();
    tr.push(&mut model, &pcm)?;
    tr.flush(&mut model)?;
    let elapsed = t0.elapsed();
    eprintln!(
        "transcribe {:.2}s audio in {elapsed:?} ({:.2}x realtime, mode {LATENCY_MODE:?})",
        pcm.len() as f64 / SAMPLE_RATE as f64,
        pcm.len() as f64 / SAMPLE_RATE as f64 / elapsed.as_secs_f64()
    );
    println!("\n{}", tr.text(&model));
    Ok(())
}
