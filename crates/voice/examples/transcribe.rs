use std::sync::mpsc::channel;

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::whisper::{self as mwhisper};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: cargo run --example transcribe -- <input.wav>");
        std::process::exit(1);
    }
    let path = &args[1];
    println!("Load WAV: {}", path);
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect(),
    };
    let mono: Vec<f32> = if channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };
    let pcm_16k = if sample_rate != 16000 {
        println!("Resample from {} Hz to 16000 Hz...", sample_rate);
        let ratio = 16000.0 / sample_rate as f32;
        let out_len = (mono.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(mono.len() - 1);
            let frac = src_idx - idx0 as f32;
            resampled.push(mono[idx0] * (1.0 - frac) + mono[idx1] * frac);
        }
        resampled
    } else {
        mono
    };
    println!("Load model...");
    let (tx, _rx) = channel();
    let model_cache_dir = paths::data_dir().join("models").join("whisper");
    std::fs::create_dir_all(&model_cache_dir)?;
    let mut lm = voice::load_model(&model_cache_dir, &tx)?;
    println!("Audio processing...");
    let mel =
        mwhisper::audio::pcm_to_mel(&lm.config, &pcm_16k, &lm.mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, lm.config.num_mel_bins, mel_len / lm.config.num_mel_bins),
        &lm.device,
    )?;
    println!("Speech recognition...\n");
    let text = lm.decoder.decode(&mel)?;
    println!("Result:");
    println!("{}", text);
    Ok(())
}
