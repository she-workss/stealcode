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
    GreedyDecoder, Nemotron, Token, streaming::StreamingEncoder,
};

const SAMPLE_RATE: u32 = 16_000;
/// One chunk = att_right+1 = 4 encoder frames = 32 mel frames = 320 ms.
const CHUNK_MEL: usize = 32;
/// Process each 320 ms chunk as soon as it arrives (the encoder caches
/// per-frame state, so per-chunk batches cost about the same in total
/// as one big batch, but text streams with minimal latency).
const BATCH_CHUNKS: usize = 8;
const BATCH_MEL: usize = CHUNK_MEL * BATCH_CHUNKS;

/// Carried RNN-T predictor state. Tokens emitted so far are committed;
/// the state is a deterministic function of the token sequence, so
/// continuing it across chunks is exact (no re-decoding).
struct StreamState {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    nh: Vec<Vec<f32>>,
    nc: Vec<Vec<f32>>,
    embed_x: Vec<f32>,
    last_token: i32,
}

impl StreamState {
    fn new(decoder: &GreedyDecoder) -> Self {
        let hdim = decoder.predictor.hidden;
        let n = decoder.predictor.layers.len();
        Self {
            h: vec![vec![0.0; hdim]; n],
            c: vec![vec![0.0; hdim]; n],
            nh: vec![vec![0.0; hdim]; n],
            nc: vec![vec![0.0; hdim]; n],
            embed_x: vec![0.0; hdim],
            last_token: -1,
        }
    }
}

/// Greedy RNN-T decode of `n` consecutive encoder frames with a carried
/// state. Appends emitted tokens to `tokens`; returns how many were
/// emitted. Mirrors `GreedyDecoder::decode` (same predictor/joint loop).
fn decode_frames(
    model: &mut Nemotron,
    frames: &[f32],
    n: usize,
    st: &mut StreamState,
    tokens: &mut Vec<Token>,
    scratch: &mut Vec<f32>,
) -> Result<usize> {
    let d_enc = model.decoder.joint.d_enc;
    let joint_h = model.decoder.joint.joint_h;
    let n_cls = model.decoder.joint.n_cls;
    let blank = model.decoder.blank_id as usize;
    let n_layers = model.decoder.predictor.layers.len();

    let xt = transpose(frames, n, d_enc);
    let mut y = Vec::new();
    model.decoder.joint.enc.forward_t(scratch, &xt, n, &mut y);
    let mut enc_proj = vec![0.0f32; n * joint_h];
    for t in 0..n {
        for j in 0..joint_h {
            enc_proj[t * joint_h + j] = y[j * n + t];
        }
    }

    let mut pred_proj = vec![0.0f32; joint_h];
    let mut summed = vec![0.0f32; joint_h];
    let mut logits = vec![0.0f32; n_cls];
    let max_iters = 16 * n + 64;
    let mut step = 0usize;
    let mut new_symbols = 0usize;
    let mut predictor_dirty = true;
    let mut iter = 0usize;
    let mut emitted = 0usize;

    while step < n && iter < max_iters {
        iter += 1;
        if predictor_dirty {
            model.decoder.predictor.step(
                st.last_token,
                &st.h,
                &st.c,
                &mut st.nh,
                &mut st.nc,
                &mut st.embed_x,
            );
            predictor_dirty = false;
        }
        let dec = &st.nh[n_layers - 1];
        model.decoder.joint.pred.matvec(dec, &mut pred_proj);
        let e = &enc_proj[step * joint_h..(step + 1) * joint_h];
        for j in 0..joint_h {
            summed[j] = (e[j] + pred_proj[j]).max(0.0); // relu
        }
        model.decoder.joint.out.matvec(&summed, &mut logits);
        let mut best = 0usize;
        let mut best_v = logits[0];
        for i in 1..n_cls {
            if logits[i] > best_v {
                best_v = logits[i];
                best = i;
            }
        }
        if best == blank {
            step += 1;
            new_symbols = 0;
        } else {
            tokens.push(Token {
                id: best as u32,
                p: 1.0,
                step,
            });
            emitted += 1;
            st.last_token = best as i32;
            std::mem::swap(&mut st.h, &mut st.nh);
            std::mem::swap(&mut st.c, &mut st.nc);
            predictor_dirty = true;
            new_symbols += 1;
            if model.decoder.max_symbols > 0
                && new_symbols >= model.decoder.max_symbols
            {
                step += 1;
                new_symbols = 0;
            }
        }
    }
    if iter >= max_iters {
        bail!("rnnt decode: iteration cap hit");
    }
    Ok(emitted)
}

/// Incremental mel frontend: frames are computed only once their STFT
/// window is fully inside the signal. Frame t needs pcm up to
/// t*hop+half; mel values match the offline `frontend.compute`
/// exactly. With `fin` set (flush) the last 2 frames, whose windows
/// hang into the right zero-pad, are computed too.
fn grow_mel(
    model: &mut Nemotron,
    pcm: &[f32],
    mel: &mut Vec<f32>,
    mel_next: &mut usize,
    fin: bool,
) {
    let pp = &model.cfg.preprocessor;
    let (hop, half, n_mels) = (pp.hop, pp.n_fft / 2, pp.n_mels);
    if pcm.len() <= half {
        return;
    }
    let stable = if fin {
        pcm.len() / hop + 1
    } else {
        (pcm.len() - half) / hop + 1
    };
    if stable <= *mel_next {
        return;
    }
    // STFT windows are [t*hop - half, t*hop + half), and a slice's
    // first 2.5 frames hang into its own left zero-pad, so start the
    // slice 3 frames early and drop those (they are already cached).
    let slice_start = (*mel_next).saturating_sub(3) * hop;
    let m = model
        .frontend
        .compute(&pcm[slice_start..])
        .unwrap_or_default();
    let avail = m.len() / n_mels;
    let take = (stable - *mel_next).min(avail.saturating_sub(3));
    let skip = (*mel_next).min(3);
    if take > 0 {
        mel.extend_from_slice(&m[skip * n_mels..(skip + take) * n_mels]);
        *mel_next += take;
    }
}

/// Incremental transcribe: encodes only the newly arrived audio,
/// decodes it and appends the new text to the line as it goes.
struct LiveTranscriber<'a> {
    model: &'a mut Nemotron,
    prompt_id: u32,
    pcm: Vec<f32>,
    mel: Vec<f32>,
    mel_next: usize,
    processed_mel: usize,
    tokens: Vec<Token>,
    st: StreamState,
    printed: usize,
    scratch: Vec<f32>,
    senc: StreamingEncoder,
}

impl<'a> LiveTranscriber<'a> {
    fn new(model: &'a mut Nemotron, prompt_id: u32) -> Self {
        let st = StreamState::new(&model.decoder);
        let senc = StreamingEncoder::new(model.encoder.cfg.n_layers);
        Self {
            model,
            prompt_id,
            pcm: Vec::new(),
            mel: Vec::new(),
            mel_next: 0,
            processed_mel: 0,
            tokens: Vec::new(),
            st,
            printed: 0,
            scratch: Vec::new(),
            senc,
        }
    }

    /// Append 16 kHz mono PCM; grows mel and processes complete batches.
    fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.pcm.extend_from_slice(samples);
        grow_mel(
            self.model,
            &self.pcm,
            &mut self.mel,
            &mut self.mel_next,
            false,
        );
        while self.mel_next - self.processed_mel >= BATCH_MEL {
            self.process(BATCH_MEL)?;
        }
        Ok(())
    }

    /// Encode+decode `new_mel` frames of new audio. The streaming
    /// encoder caches every frame, so the encoder output of the new
    /// frames is identical to an offline full-audio encode, and the
    /// committed prefix is stable.
    fn process(&mut self, new_mel: usize) -> Result<()> {
        let n_mels = self.model.cfg.preprocessor.n_mels;
        let t0_mel = self.processed_mel;
        self.senc.encode_new(
            &mut self.model.encoder,
            &self.mel,
            n_mels,
            t0_mel,
            t0_mel + new_mel,
            Some(self.prompt_id),
        )?;
        let s = t0_mel / 8;
        let count = new_mel.div_ceil(8);
        let frames = self.senc.frames(s, s + count)?;
        decode_frames(
            self.model,
            frames,
            count,
            &mut self.st,
            &mut self.tokens,
            &mut self.scratch,
        )?;
        self.processed_mel += new_mel;
        self.print_live();
        Ok(())
    }

    /// End of stream: encode+decode the leftover tail (< BATCH_MEL).
    fn flush(&mut self) -> Result<()> {
        grow_mel(
            self.model,
            &self.pcm,
            &mut self.mel,
            &mut self.mel_next,
            true,
        );
        let total = self
            .model
            .frontend
            .n_frames(self.pcm.len())
            .min(self.mel_next);
        let tail = total.saturating_sub(self.processed_mel);
        if tail > 0 {
            self.process(tail)?;
        }
        self.print_live();
        Ok(())
    }

    /// Print only the newly appended transcript text (the committed
    /// prefix is never re-printed or rewritten).
    fn print_live(&mut self) {
        let ids: Vec<u32> = self.tokens.iter().map(|t| t.id).collect();
        let text = self.model.tokenizer.decode_transcript(&ids, true);
        if text.len() > self.printed {
            print!("{}", &text[self.printed..]);
            io::stdout().flush().ok();
            self.printed = text.len();
        }
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
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
    let timing = std::env::var_os("STEALCODE_TIMING").is_some();
    while off < pcm.len() {
        let end = (off + chunk_samples).min(pcm.len());
        let t0 = Instant::now();
        tr.push(&pcm[off..end])?;
        if timing {
            eprintln!(
                "[demo] push {:.3}s audio -> {:.3}s (mel {}, tokens {})",
                (end - off) as f64 / SAMPLE_RATE as f64,
                t0.elapsed().as_secs_f64(),
                tr.mel_next,
                tr.tokens.len()
            );
        }
        off = end;
        if off < pcm.len() {
            std::thread::sleep(Duration::from_millis(chunk_ms as u64));
        }
    }
    let t0 = Instant::now();
    tr.flush()?;
    if timing {
        eprintln!("[demo] flush: {:.3}s", t0.elapsed().as_secs_f64());
    }
    let ids: Vec<u32> = tr.tokens.iter().map(|t| t.id).collect();
    let text = tr.model.tokenizer.decode_transcript(&ids, true);
    println!("\n\n[FINAL] {text}");
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
    loop {
        let samples: Vec<f32> = {
            let mut b = buf.lock().unwrap();
            std::mem::take(&mut *b)
        };
        if !samples.is_empty() {
            tr.push(&samples)?;
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
        tr.push(&samples)?;
    }
    tr.flush()?;
    let ids: Vec<u32> = tr.tokens.iter().map(|t| t.id).collect();
    let text = tr.model.tokenizer.decode_transcript(&ids, true);
    println!("\n\n[FINAL] {text}");
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
