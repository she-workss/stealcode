//! App-facing speech-to-text service.
//!
//! Wraps the `audio_engine` crate (Nemotron model) behind a
//! microphone-capturing worker: [`VoiceManager::toggle`] records,
//! partial transcripts stream out as [`VoiceEvent::Partial`], and the
//! final text arrives as [`VoiceEvent::Transcribed`]. The model
//! auto-downloads on first use (see [`models`]).

pub mod models;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
    },
    time::{Duration, Instant},
};

use audio_engine::{
    Nemotron,
    dsp::to_mono_16k,
    model::{AsrModel, LatencyMode, LiveAsr},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tracing::{error, info};

const TICK: Duration = Duration::from_millis(200);
const MIN_PARTIAL_SECS: f32 = 0.8;
const PARTIAL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone)]
pub enum VoiceEvent {
    Status(String),
    /// Partial recognition result while recording.
    Partial(String),
    Transcribed(String),
    Error(String),
}

#[derive(Debug, Clone)]
enum VoiceCommand {
    Start,
    Stop,
}

#[derive(Debug)]
enum TranscribeJob {
    Partial(Vec<f32>),
    Final(Vec<f32>),
    /// Load the model only (sets model_ready), no transcription. Sent on
    /// Start so partial results arrive in real time.
    Warmup,
    /// Release the model; RAM returns to baseline after the final transcript.
    Unload,
}

#[derive(Debug)]
pub struct VoiceManager {
    tx_cmd: Option<Sender<VoiceCommand>>,
    rx_event: Option<Receiver<VoiceEvent>>,
    pub is_recording: bool,
    pub text: String,
    pub status: String,
}

impl VoiceManager {
    pub fn new() -> Self {
        Self {
            tx_cmd: None,
            rx_event: None,
            is_recording: false,
            text: String::new(),
            status: "Waiting (Ctrl+G)".to_string(),
        }
    }

    fn start_worker_if_needed(&mut self) {
        if self.tx_cmd.is_some() {
            return;
        }
        let (tx_cmd, rx_cmd) = std::sync::mpsc::channel();
        let (tx_event, rx_event) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            voice_worker(rx_cmd, tx_event);
        });
        self.tx_cmd = Some(tx_cmd);
        self.rx_event = Some(rx_event);
    }

    pub fn toggle(&mut self) {
        self.start_worker_if_needed();
        if let Some(tx) = &self.tx_cmd {
            if self.is_recording {
                self.status = "Transcribing...".to_string();
                let _ = tx.send(VoiceCommand::Stop);
                self.is_recording = false;
            } else {
                self.status = "Recording...".to_string();
                self.text.clear();
                let _ = tx.send(VoiceCommand::Start);
                self.is_recording = true;
            }
        }
    }

    pub fn poll_events(&mut self) {
        if let Some(rx) = &self.rx_event {
            loop {
                match rx.try_recv() {
                    Ok(event) => match event {
                        VoiceEvent::Status(s) => self.status = s,
                        VoiceEvent::Partial(t) => self.text = t,
                        VoiceEvent::Transcribed(t) => {
                            self.text = t;
                            self.status = "Ready (Ctrl+G)".to_string();
                        }
                        VoiceEvent::Error(e) => {
                            self.status = format!("Error: {}", e);
                        }
                    },
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        // Worker exited; clear channels
                        self.tx_cmd = None;
                        self.rx_event = None;
                        break;
                    }
                }
            }
        }
    }
}

fn voice_worker(rx_cmd: Receiver<VoiceCommand>, tx_event: Sender<VoiceEvent>) {
    let host = cpal::default_host();
    let audio_device = match host.default_input_device() {
        Some(d) => d,
        None => {
            let _ =
                tx_event.send(VoiceEvent::Error("No microphone".to_string()));
            return;
        }
    };
    let supported_config = match audio_device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx_event
                .send(VoiceEvent::Error(format!("Audio config: {:?}", e)));
            return;
        }
    };
    let stream_config: cpal::StreamConfig = supported_config.clone().into();
    let sample_rate = stream_config.sample_rate;
    let channels = stream_config.channels as usize;
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    // The model loads at worker start; partials only run once ready
    // (otherwise snapshots would be stale).
    let model_ready = Arc::new(AtomicBool::new(false));
    // Transcribe thread: owns the model, processes jobs in order.
    let (tx_job, rx_job) = std::sync::mpsc::sync_channel::<TranscribeJob>(1);
    let (tx_done, rx_done) = std::sync::mpsc::channel::<()>();
    let tx_event_worker = tx_event.clone();
    let model_ready_worker = Arc::clone(&model_ready);
    std::thread::spawn(move || {
        transcribe_worker(rx_job, tx_event_worker, tx_done, model_ready_worker)
    });
    let mut stream: Option<cpal::Stream> = None;
    let mut last_partial = Instant::now();
    let mut ready_announced = false;
    // Interleaved samples already handed to the transcriber; each job only
    // carries new samples (no tail re-encoding).
    let mut sent_samples: usize = 0;
    loop {
        let cmd = if stream.is_some() {
            rx_cmd.recv_timeout(TICK)
        } else {
            match rx_cmd.recv() {
                Ok(c) => Ok(c),
                Err(_) => break,
            }
        };
        match cmd {
            Ok(VoiceCommand::Start) => {
                if stream.is_none() {
                    if let Ok(mut buf) = audio_buffer.lock() {
                        buf.clear();
                    }
                    sent_samples = 0;
                    let buf_clone = Arc::clone(&audio_buffer);
                    let stream_result = match supported_config.sample_format() {
                        cpal::SampleFormat::F32 => audio_device
                            .build_input_stream(
                            stream_config.clone(),
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                if let Ok(mut buf) = buf_clone.lock() {
                                    buf.extend_from_slice(data);
                                }
                            },
                            |e| error!("Audio stream error: {:?}", e),
                            None,
                        ),
                        _ => continue,
                    };
                    if let Ok(s) = stream_result {
                        if s.play().is_ok() {
                            stream = Some(s);
                            last_partial = Instant::now();
                            ready_announced = false;
                            let _ = tx_event.send(VoiceEvent::Status(
                                "Recording...".to_string(),
                            ));
                        }
                    }
                }
                // Warm the model at Start so partials stream in real time, not
                // only at Stop.
                let _ = tx_job.try_send(TranscribeJob::Warmup);
            }
            Ok(VoiceCommand::Stop) => {
                drop(stream.take());
                let raw_audio = audio_buffer
                    .lock()
                    .map(|mut buf| std::mem::take(&mut *buf))
                    .unwrap_or_default();
                if !raw_audio.is_empty() {
                    // Transcriber has everything up to sent_samples; send the
                    // un-sent tail, then flush.
                    let new_pcm_16k = to_mono_16k(
                        &raw_audio[sent_samples.min(raw_audio.len())..],
                        channels,
                        sample_rate,
                    );
                    if tx_job.send(TranscribeJob::Final(new_pcm_16k)).is_ok() {
                        // Wait for the final transcript, then unload the model
                        // (RAM drops to baseline; the
                        // next recording reloads it).
                        let _ = rx_done.recv();
                        let _ = tx_job.send(TranscribeJob::Unload);
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !ready_announced && model_ready.load(Ordering::Relaxed) {
                    ready_announced = true;
                    let _ = tx_event
                        .send(VoiceEvent::Status("Recording...".to_string()));
                }
                let buf_len =
                    audio_buffer.lock().map(|buf| buf.len()).unwrap_or(0);
                let secs =
                    buf_len as f32 / sample_rate as f32 / channels as f32;
                if model_ready.load(Ordering::Relaxed)
                    && secs >= MIN_PARTIAL_SECS
                    && last_partial.elapsed() >= PARTIAL_INTERVAL
                {
                    let Ok(raw) = audio_buffer.lock() else {
                        continue;
                    };
                    if raw.len() > sent_samples {
                        // Send only new samples; the streaming encoder extends
                        // incrementally.
                        let pcm_16k = to_mono_16k(
                            &raw[sent_samples..],
                            channels,
                            sample_rate,
                        );
                        // sync_channel(1): try_send fails while the transcriber
                        // is busy - skip the update,
                        // sent_samples stays put, the tail goes next time or at
                        // Stop.
                        if tx_job
                            .try_send(TranscribeJob::Partial(pcm_16k))
                            .is_ok()
                        {
                            sent_samples = raw.len();
                            last_partial = Instant::now();
                        }
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn transcribe_worker(
    rx_job: Receiver<TranscribeJob>,
    tx_event: Sender<VoiceEvent>,
    tx_done: Sender<()>,
    model_ready: Arc<AtomicBool>,
) {
    let mut model: Option<Nemotron> = None;
    // The streaming transcriber keeps state separate from the model, so both
    // can live side by side.
    let mut live: Option<Box<dyn LiveAsr<Nemotron>>> = None;
    while let Ok(job) = rx_job.recv() {
        match job {
            TranscribeJob::Unload => {
                live = None;
                model = None;
                model_ready.store(false, Ordering::Relaxed);
                info!("model unloaded (RAM released)");
            }
            TranscribeJob::Warmup => {
                // Warmup: load the model if absent, wait for the next job.
                if model.is_none() {
                    model = load_into(&tx_event, &model_ready);
                }
            }
            TranscribeJob::Partial(pcm) => {
                process_stream(
                    &mut model,
                    &mut live,
                    &tx_event,
                    &model_ready,
                    &pcm,
                    false,
                );
            }
            TranscribeJob::Final(pcm) => {
                process_stream(
                    &mut model,
                    &mut live,
                    &tx_event,
                    &model_ready,
                    &pcm,
                    true,
                );
                let _ = tx_done.send(());
            }
        }
    }
}

/// Push new samples through the streaming transcriber (lazily loading
/// the model and stream state) and publish an event; `final_job` flushes.
fn process_stream(
    model: &mut Option<Nemotron>,
    live: &mut Option<Box<dyn LiveAsr<Nemotron>>>,
    tx_event: &Sender<VoiceEvent>,
    model_ready: &Arc<AtomicBool>,
    pcm: &[f32],
    final_job: bool,
) {
    if model.is_none() {
        *model = load_into(tx_event, model_ready);
    }
    let Some(m) = model.as_mut() else { return };
    if live.is_none() {
        match m.live(LatencyMode::Standard) {
            Ok(tr) => *live = Some(tr),
            Err(e) => {
                let _ = tx_event
                    .send(VoiceEvent::Error(format!("Session init: {e:?}")));
                return;
            }
        }
    }
    let Some(tr) = live.as_mut() else { return };
    let t_decode = Instant::now();
    let result = (|| -> anyhow::Result<String> {
        tr.push(m, pcm)?;
        if final_job {
            tr.flush(m)?;
        }
        Ok(tr.text(m))
    })();
    match result {
        Ok(text) => {
            info!(
                "Decode voice message (final={final_job}): {:?}",
                t_decode.elapsed()
            );
            let event = if final_job {
                VoiceEvent::Transcribed(text)
            } else {
                VoiceEvent::Partial(text)
            };
            let _ = tx_event.send(event);
        }
        Err(e) => {
            let _ = tx_event
                .send(VoiceEvent::Error(format!("Decode error: {e:?}")));
        }
    }
}

/// Load the model (auto-downloading it on first use), publishing status
/// events; None on error.
fn load_into(
    tx_event: &Sender<VoiceEvent>,
    model_ready: &Arc<AtomicBool>,
) -> Option<Nemotron> {
    let _ = tx_event.send(VoiceEvent::Status("Loading model...".to_string()));
    let mut next_report = 8u64 * 1024 * 1024;
    match models::load_model(|done, total| {
        if done < next_report && total.is_none_or(|t| done != t) {
            return;
        }
        next_report = done + 8 * 1024 * 1024;
        let status = match total {
            Some(total) if total > 0 => format!(
                "Downloading model... {} MB / {} MB ({:.0}%)",
                done / (1024 * 1024),
                total / (1024 * 1024),
                done as f64 * 100.0 / total as f64
            ),
            _ => format!("Downloading model... {} MB", done / (1024 * 1024)),
        };
        let _ = tx_event.send(VoiceEvent::Status(status));
    }) {
        Ok(m) => {
            model_ready.store(true, Ordering::Relaxed);
            Some(m)
        }
        Err(e) => {
            let _ =
                tx_event.send(VoiceEvent::Error(format!("Model load: {e:?}")));
            model_ready.store(false, Ordering::Relaxed);
            None
        }
    }
}
