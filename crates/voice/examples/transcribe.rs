//! Offline transcription of a 16 kHz mono WAV via the nemotron port.
//!
//! Usage: cargo run -p voice --example transcribe -- <model.gguf> <audio.wav>

use std::path::Path;

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        voice::default_model_path().to_string_lossy().into_owned()
    });
    let wav_path = args
        .next()
        .with_context(|| "usage: transcribe <model.gguf> <audio.wav>")?;

    let mut model = voice::nemotron::Nemotron::load(Path::new(&model_path))?;

    let mut reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("expected mono, got {} ch", spec.channels);
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
                b => anyhow::bail!("unsupported bits_per_sample {b}"),
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let pcm = if spec.sample_rate != 16000 {
        eprintln!("resampling {} Hz -> 16 kHz", spec.sample_rate);
        let ratio = 16000.0 / spec.sample_rate as f32;
        let out_len = (pcm.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(pcm.len() - 1);
            let frac = src_idx - idx0 as f32;
            resampled.push(pcm[idx0] * (1.0 - frac) + pcm[idx1] * frac);
        }
        resampled
    } else {
        pcm
    };
    eprintln!(
        "{} samples ({:.2} s), {} mel frames",
        pcm.len(),
        pcm.len() as f64 / 16000.0,
        model.frontend.n_frames(pcm.len())
    );

    let t0 = std::time::Instant::now();
    let raw = model.transcribe_raw(&pcm, None)?;
    let ids: Vec<u32> = raw.iter().map(|t| t.id).collect();
    let text = model.tokenizer.decode_transcript(&ids, true);
    let elapsed = t0.elapsed();
    eprintln!(
        "transcribe {:.2}s аудио: {elapsed:?} ({:.2}x realtime)",
        pcm.len() as f64 / 16000.0,
        pcm.len() as f64 / 16000.0 / elapsed.as_secs_f64()
    );
    println!("{}", text);
    Ok(())
}
