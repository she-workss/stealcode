//! Live Audio Transcription — Foundry Local Rust SDK style.
//!
//! Tries CPAL mic capture first; falls back to a synthetic 440 Hz sine
//! if no input device is available. The transcript is printed inline
//! as chunks are recognized (committed prefix is stable — later audio
//! never rewrites earlier text), and `[FINAL]` at the end.
//!
//! Usage:
//!   cargo run -p voice --example live_audio_transcription            # Live
//! mic (Ctrl+C / Enter to stop)   cargo run -p voice --example
//! live_audio_transcription -- --synth  # Synthetic 440Hz sine wave   cargo run
//! -p voice --example live_audio_transcription -- --wav audio.wav   cargo run
//! -p voice --example live_audio_transcription -- --model path/to/model.gguf

use std::{
    io::{self, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use voice::nemotron::{
    Nemotron,
    live::{CHUNK_MEL, LiveTranscriber},
};

const SAMPLE_RATE: u32 = 16_000;

/// Print only the newly appended transcript text (the committed prefix
/// is never re-printed or rewritten).
fn print_new(model: &Nemotron, tr: &LiveTranscriber, printed: &mut usize) {
    let text = tr.text(model);
    if text.len() > *printed {
        print!("{}", &text[*printed..]);
        io::stdout().flush().ok();
        *printed = text.len();
    }
}

/// Replay pre-recorded PCM in real-time 320 ms chunks.
#[allow(clippy::too_many_lines)]
fn run_bench(model: &Nemotron) {
    let d = model.encoder.cfg.d_model;
    let mut scratch = Vec::new();
    let x = vec![0.1f32; d * 32];
    let lin = &model.encoder.blocks[0].ff1_lin1;
    let t0 = std::time::Instant::now();
    let mut w = Vec::new();
    lin.q.as_ref().unwrap().to_f32(&mut w);
    let deq_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut y = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        lin.forward_t(&mut scratch, &x, 32, &mut y);
    }
    let fwd_ms = t0.elapsed().as_secs_f64() / 10.0 * 1e3;
    let copy = w.to_vec();
    let mut y2 = vec![0.0f32; lin.out * 32];
    let t0 = std::time::Instant::now();
    unsafe {
        matrixmultiply::sgemm(
            lin.out,
            lin.inp,
            32,
            1.0,
            w.as_ptr(),
            lin.inp as isize,
            1,
            x.as_ptr(),
            32,
            1,
            0.0,
            y2.as_mut_ptr(),
            32,
            1,
        );
    }
    let w_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    unsafe {
        matrixmultiply::sgemm(
            lin.out,
            lin.inp,
            32,
            1.0,
            copy.as_ptr(),
            lin.inp as isize,
            1,
            x.as_ptr(),
            32,
            1,
            0.0,
            y2.as_mut_ptr(),
            32,
            1,
        );
    }
    let copy_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "STEALCODE_BENCH: ff1_lin1 {}x{} deq={:.1}ms fwd_t={:.2}ms sgemm(deq)={:.2}ms sgemm(copy)={:.2}ms",
        lin.out, lin.inp, deq_ms, fwd_ms, w_ms, copy_ms
    );
    // simulate the encode pattern: fresh Vec outputs per call
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        let mut y3 = Vec::new();
        lin.forward_t(&mut scratch, &x, 32, &mut y3);
    }
    let alloc_ms = t0.elapsed().as_secs_f64() / 10.0 * 1e3;
    println!("STEALCODE_BENCH: with fresh Vec per call: {alloc_ms:.2}ms");
}

fn run_source(
    model: &mut Nemotron,
    prompt_id: u32,
    pcm: &[f32],
    label: &str,
) -> Result<()> {
    let pp = &model.cfg.preprocessor;
    let chunk_samples = CHUNK_MEL * pp.hop;
    let chunk_ms = chunk_samples * 1000 / pp.sample_rate;
    println!(
        "Replaying {label} ({:.2}s) in {chunk_ms} ms chunks...\n",
        pcm.len() as f64 / SAMPLE_RATE as f64
    );
    let mut tr = LiveTranscriber::new(model, prompt_id);
    let mut off = 0;
    let mut printed = 0usize;
    let timing = std::env::var_os("STEALCODE_TIMING").is_some();
    while off < pcm.len() {
        let end = (off + chunk_samples).min(pcm.len());
        let t0 = Instant::now();
        tr.push(model, &pcm[off..end])?;
        print_new(model, &tr, &mut printed);
        if timing {
            eprintln!(
                "[demo] push {:.3}s audio -> {:.3}s (mel {}, tokens {})",
                (end - off) as f64 / SAMPLE_RATE as f64,
                t0.elapsed().as_secs_f64(),
                tr.mel_next(),
                tr.tokens().len()
            );
        }
        off = end;
        if off < pcm.len() {
            std::thread::sleep(Duration::from_millis(chunk_ms as u64));
        }
    }
    let t0 = Instant::now();
    tr.flush(model)?;
    print_new(model, &tr, &mut printed);
    if timing {
        eprintln!("[demo] flush: {:.3}s", t0.elapsed().as_secs_f64());
    }
    println!("\n\n[FINAL] {}", tr.text(model));
    Ok(())
}

/// Live microphone capture (CPAL) with a "press Enter to stop" loop.
fn run_mic(model: &mut Nemotron, prompt_id: u32) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no input audio device available")?;
    let default_config = device.default_input_config()?;
    let device_rate = default_config.sample_rate();
    let device_channels = default_config.channels();
    let sample_format = default_config.sample_format();

    let mic_config = cpal::StreamConfig {
        channels: device_channels,
        sample_rate: device_rate,
        buffer_size: cpal::BufferSize::Default,
    };
    // Unbounded shared buffer: the capture callback never blocks and no
    // audio is ever dropped, even while a batch is being encoded.
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let err_fn = |e| eprintln!("Microphone stream error: {e:?}");

    let input_stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let v = to_16k_mono(data, device_channels, device_rate);
                    if !v.is_empty() {
                        buf.lock().unwrap().extend_from_slice(&v);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let s: Vec<f32> = data
                        .iter()
                        .map(|&x| x as f32 / i16::MAX as f32)
                        .collect();
                    let v = to_16k_mono(&s, device_channels, device_rate);
                    if !v.is_empty() {
                        buf.lock().unwrap().extend_from_slice(&v);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let s: Vec<f32> = data
                        .iter()
                        .map(|&x| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect();
                    let v = to_16k_mono(&s, device_channels, device_rate);
                    if !v.is_empty() {
                        buf.lock().unwrap().extend_from_slice(&v);
                    }
                },
                err_fn,
                None,
            )?
        }
        other => bail!("unsupported input sample format: {other:?}"),
    };
    input_stream.play()?;

    println!("===========================================================");
    println!("  LIVE TRANSCRIPTION ACTIVE");
    println!("  Speak into your microphone. Press Enter to stop.");
    println!("===========================================================");
    println!();

    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = Arc::clone(&stop);
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        stop2.store(true, Ordering::SeqCst);
    });

    let mut tr = LiveTranscriber::new(model, prompt_id);
    let mut printed = 0usize;
    loop {
        let samples: Vec<f32> = {
            let mut b = buf.lock().unwrap();
            std::mem::take(&mut *b)
        };
        if !samples.is_empty() {
            tr.push(model, &samples)?;
            print_new(model, &tr, &mut printed);
        }
        if stop.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    drop(input_stream);
    let samples: Vec<f32> = {
        let mut b = buf.lock().unwrap();
        std::mem::take(&mut *b)
    };
    if !samples.is_empty() {
        tr.push(model, &samples)?;
        print_new(model, &tr, &mut printed);
    }
    tr.flush(model)?;
    print_new(model, &tr, &mut printed);
    println!("\n\n[FINAL] {}", tr.text(model));
    Ok(())
}

/// Mix down to mono and resample to 16 kHz (linear).
fn to_16k_mono(data: &[f32], channels: u16, sample_rate: u32) -> Vec<f32> {
    let mono: Vec<f32> = if channels > 1 {
        data.chunks(channels as usize)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        data.to_vec()
    };
    if sample_rate == SAMPLE_RATE {
        mono
    } else {
        let ratio = SAMPLE_RATE as f32 / sample_rate as f32;
        let out_len = (mono.len() as f32 * ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f32 / ratio;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(mono.len() - 1);
            let frac = src - i0 as f32;
            out.push(mono[i0] * (1.0 - frac) + mono[i1] * frac);
        }
        out
    }
}

/// 16 kHz mono f32 PCM (440 Hz sine: exercises the full pipeline
/// without a microphone).
fn generate_sine(sample_rate: usize, seconds: usize, freq: f64) -> Vec<f32> {
    let n = sample_rate * seconds;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            0.5 * (2.0 * std::f64::consts::PI * freq * t).sin() as f32
        })
        .collect()
}

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
    let pcm = if spec.sample_rate != SAMPLE_RATE {
        to_16k_mono(&pcm, 1, spec.sample_rate)
    } else {
        pcm
    };
    Ok(pcm)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    let mut wav_path: Option<String> = None;
    let mut synth = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next(),
            "--wav" => wav_path = args.next(),
            "--synth" => synth = true,
            other => bail!("unknown arg {other}"),
        }
    }
    let model_path = model_path.unwrap_or_else(|| {
        voice::default_model_path().to_string_lossy().into_owned()
    });

    println!("===========================================================");
    println!("  Live Audio Transcription Demo (Rust)");
    println!("===========================================================");
    println!();
    println!("Loading model...");
    let mut model = Nemotron::load(std::path::Path::new(&model_path))?;
    let prompt_id = model.resolve_prompt_id(None)?;
    println!("✓ Model loaded\n");

    if std::env::var_os("STEALCODE_BENCH").is_some() {
        run_bench(&model);
    }

    if let Some(wav) = wav_path {
        let pcm = read_wav_16k(&wav)?;
        let r = run_source(&mut model, prompt_id, &pcm, &wav);
        if std::env::var_os("STEALCODE_BENCH").is_some() {
            run_bench(&model);
        }
        return r;
    }
    if synth {
        let pcm = generate_sine(SAMPLE_RATE as usize, 2, 440.0);
        return run_source(
            &mut model,
            prompt_id,
            &pcm,
            "synthetic 440 Hz sine",
        );
    }
    match run_mic(&mut model, prompt_id) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Could not initialize microphone: {e}");
            eprintln!("Falling back to synthetic audio test...\n");
            let pcm = generate_sine(SAMPLE_RATE as usize, 2, 440.0);
            run_source(&mut model, prompt_id, &pcm, "synthetic 440 Hz sine")
        }
    }
}
