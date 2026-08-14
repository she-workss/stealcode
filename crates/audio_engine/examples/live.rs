//! Real-time speech-to-text from the microphone (CPAL).
//!
//! The transcript is printed inline as chunks are recognized.
//!
//! Usage:
//!   cargo run -p audio_engine --example live -- --model path/to/model.gguf

use std::{
    io::{self, Write},
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use audio_engine::{
    Nemotron,
    dsp::to_mono_16k,
    model::{AsrModel, LatencyMode, LiveAsr},
};
use cpal::{
    BufferSize, InputCallbackInfo, SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

/// How much audio a streaming encoder step covers, trading latency
/// against throughput.
const LATENCY_MODE: LatencyMode = LatencyMode::Medium;

/// Print only the newly appended transcript text (the committed prefix
/// is never re-printed or rewritten).
fn print_new(
    model: &Nemotron,
    tr: &dyn LiveAsr<Nemotron>,
    printed: &mut usize,
) {
    let text = tr.text(model);
    if text.len() > *printed {
        print!("{}", &text[*printed..]);
        io::stdout().flush().ok();
        *printed = text.len();
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next(),
            other => bail!("unknown arg {other}"),
        }
    }
    let model_path =
        model_path.with_context(|| "usage: live --model <model.gguf>")?;
    println!("Loading model...");
    let mut model = Nemotron::load(Path::new(&model_path))?;
    println!("Model loaded\n");
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no input audio device available")?;
    let default_config = device.default_input_config()?;
    let device_rate = default_config.sample_rate();
    let device_channels = default_config.channels();
    let sample_format = default_config.sample_format();

    let mic_config = StreamConfig {
        channels: device_channels,
        sample_rate: device_rate,
        buffer_size: BufferSize::Default,
    };
    // Unbounded shared buffer: the capture callback never blocks and no
    // audio is ever dropped, even while a batch is being encoded.
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let err_fn = |e| eprintln!("Microphone stream error: {e:?}");
    let input_stream = match sample_format {
        SampleFormat::F32 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[f32], _: &InputCallbackInfo| {
                    let v =
                        to_mono_16k(data, device_channels.into(), device_rate);
                    if !v.is_empty() {
                        buf.lock().unwrap().extend_from_slice(&v);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[i16], _: &InputCallbackInfo| {
                    let s: Vec<f32> = data
                        .iter()
                        .map(|&x| x as f32 / i16::MAX as f32)
                        .collect();
                    let v =
                        to_mono_16k(&s, device_channels.into(), device_rate);
                    if !v.is_empty() {
                        buf.lock().unwrap().extend_from_slice(&v);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let buf = Arc::clone(&buf);
            device.build_input_stream(
                mic_config.clone(),
                move |data: &[u16], _: &InputCallbackInfo| {
                    let s: Vec<f32> = data
                        .iter()
                        .map(|&x| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .collect();
                    let v =
                        to_mono_16k(&s, device_channels.into(), device_rate);
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
    println!("Live transcription. Speak into your microphone.");
    let mut tr = model.live(LATENCY_MODE)?;
    let mut printed = 0usize;
    loop {
        let samples: Vec<f32> = {
            let mut b = buf.lock().unwrap();
            std::mem::take(&mut *b)
        };
        if !samples.is_empty() {
            tr.push(&mut model, &samples)?;
            print_new(&model, tr.as_ref(), &mut printed);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}
