pub mod nemotron;

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
    },
    time::{Duration, Instant},
};

/// Период опроса команд/таймера во время записи.
const TICK: Duration = Duration::from_millis(200);
/// Не транскрибируем меньше этого объёма накопленного аудио.
const MIN_PARTIAL_SECS: f32 = 0.8;
/// Минимальная пауза между промежуточными распознаваниями.
const PARTIAL_INTERVAL: Duration = Duration::from_millis(1500);
/// Окно скользящего хвоста для частичных распознаваний длинных записей.
const PARTIAL_WINDOW_SECS: f32 = 8.0;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tracing::{error, info};

/// Local nemotron GGUF checkpoint (overridable via
/// STEALCODE_NEMOTRON_GGUF).
pub fn default_model_path() -> std::path::PathBuf {
    match std::env::var_os("STEALCODE_NEMOTRON_GGUF") {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(
            "D:\\Programming\\stealcode\\nemotron\\nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        ),
    }
}

#[derive(Debug, Clone)]
pub enum VoiceEvent {
    Status(String),
    /// Промежуточный результат распознавания во время записи.
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
    /// Только прогреть модель (загрузить, выставить model_ready),
    /// без транскрипции. Шлём при старте записи, чтобы частичные
    /// распознавания шли в реалтайме.
    Warmup,
    /// Освободить модель: после финальной транскрипции останавливаем
    /// запись, RAM должна вернуться к базовому уровню.
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
                        // Поток завершился (return), очищаем каналы
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
                tx_event.send(VoiceEvent::Error("Нет микрофона".to_string()));
            return;
        }
    };
    let supported_config = match audio_device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx_event
                .send(VoiceEvent::Error(format!("Конфиг аудио: {:?}", e)));
            return;
        }
    };
    let stream_config: cpal::StreamConfig = supported_config.clone().into();
    let sample_rate = stream_config.sample_rate;
    let channels = stream_config.channels as usize;
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    // Модель грузится сразу при старте воркера; частичные распознавания
    // включаются только после готовности (иначе снапшоты устаревают).
    let model_ready = Arc::new(AtomicBool::new(false));

    // Поток распознавания: владеет моделью, выполняет задания по очереди.
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
                    audio_buffer.lock().unwrap().clear();
                    let buf_clone = Arc::clone(&audio_buffer);
                    let stream_result = match supported_config.sample_format() {
                        cpal::SampleFormat::F32 => audio_device
                            .build_input_stream(
                            stream_config.clone(),
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                buf_clone
                                    .lock()
                                    .unwrap()
                                    .extend_from_slice(data);
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
                                "Идет запись...".to_string(),
                            ));
                        }
                    }
                }
                // Греем модель сразу при старте записи, чтобы
                // частичные распознавания шли в реалтайме, а не
                // после Stop (иначе модель загрузится только на
                // финальной транскрипции).
                let _ = tx_job.try_send(TranscribeJob::Warmup);
            }
            Ok(VoiceCommand::Stop) => {
                drop(stream.take());
                let raw_audio = audio_buffer.lock().unwrap().clone();
                if !raw_audio.is_empty() {
                    let pcm_16k =
                        to_mono_16k(&raw_audio, channels, sample_rate);
                    if tx_job.send(TranscribeJob::Final(pcm_16k)).is_ok() {
                        // Ждём финальную транскрипцию, затем освобождаем
                        // модель (RAM падает к базовому уровню). Следующая
                        // запись снова загрузит модель (статус покажет
                        // «Загрузка модели...»).
                        let _ = rx_done.recv();
                        let _ = tx_job.send(TranscribeJob::Unload);
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !ready_announced && model_ready.load(Ordering::Relaxed) {
                    ready_announced = true;
                    let _ = tx_event
                        .send(VoiceEvent::Status("Идет запись...".to_string()));
                }
                let secs = audio_buffer.lock().unwrap().len() as f32
                    / sample_rate as f32
                    / channels as f32;
                if model_ready.load(Ordering::Relaxed)
                    && secs >= MIN_PARTIAL_SECS
                    && last_partial.elapsed() >= PARTIAL_INTERVAL
                {
                    let raw_audio = audio_buffer.lock().unwrap().clone();
                    // На длинных записях распознаём только хвост (последние N
                    // секунд).
                    let start = if secs > PARTIAL_WINDOW_SECS {
                        ((secs - PARTIAL_WINDOW_SECS)
                            * sample_rate as f32
                            * channels as f32) as usize
                    } else {
                        0
                    };
                    let pcm_16k =
                        to_mono_16k(&raw_audio[start..], channels, sample_rate);
                    // sync_channel(1): try_send падает, пока транскриптор занят
                    // (1 задание в полёте/буфере) — пропускаем обновление.
                    if tx_job.try_send(TranscribeJob::Partial(pcm_16k)).is_ok()
                    {
                        last_partial = Instant::now();
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
    let mut model: Option<nemotron::Nemotron> = None;
    while let Ok(job) = rx_job.recv() {
        match job {
            TranscribeJob::Unload => {
                model = None;
                model_ready.store(false, Ordering::Relaxed);
                info!("model unloaded (RAM released)");
                continue;
            }
            TranscribeJob::Warmup => {
                // Прогрев: загружаем модель (если её ещё нет) и ждём
                // следующего задания.
                if model.is_none() {
                    model = load_into(&tx_event, &model_ready);
                    if model.is_none() {
                        continue;
                    }
                }
                continue;
            }
            _ => {}
        }
        // Ленивая (пере)загрузка: первая транскрипция и каждая после
        // Unload снова грузят модель.
        if model.is_none() {
            model = load_into(&tx_event, &model_ready);
            if model.is_none() {
                continue;
            }
        }
        let Some(m) = model.as_mut() else { return };
        let (pcm, final_job) = match job {
            TranscribeJob::Partial(pcm) => (pcm, false),
            TranscribeJob::Final(pcm) => (pcm, true),
            TranscribeJob::Warmup | TranscribeJob::Unload => {
                unreachable!("handled above")
            }
        };
        let t_decode = Instant::now();
        match m.transcribe(&pcm, None) {
            Ok(tr) => {
                info!(
                    "Decode voice message (final={}): {:?}",
                    final_job,
                    t_decode.elapsed()
                );
                let event = if final_job {
                    VoiceEvent::Transcribed(tr.text)
                } else {
                    VoiceEvent::Partial(tr.text)
                };
                let _ = tx_event.send(event);
            }
            Err(e) => {
                let _ = tx_event
                    .send(VoiceEvent::Error(format!("Decode error: {:?}", e)));
            }
        }
        if final_job {
            let _ = tx_done.send(());
        }
    }
}

/// Загрузка модели с публикацией статусов; возвращает None при ошибке.
fn load_into(
    tx_event: &Sender<VoiceEvent>,
    model_ready: &Arc<AtomicBool>,
) -> Option<nemotron::Nemotron> {
    let _ = tx_event.send(VoiceEvent::Status("Загрузка модели...".to_string()));
    match load_model() {
        Ok(m) => {
            model_ready.store(true, Ordering::Relaxed);
            Some(m)
        }
        Err(e) => {
            let _ = tx_event
                .send(VoiceEvent::Error(format!("Model load: {:?}", e)));
            model_ready.store(false, Ordering::Relaxed);
            None
        }
    }
}

/// Приводит интерливованный буфер к моно 16 кГц.
fn to_mono_16k(
    raw_audio: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Vec<f32> {
    let mono: Vec<f32> = if channels <= 1 {
        raw_audio.to_vec()
    } else {
        raw_audio
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if sample_rate != 16000 {
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
    }
}

pub fn load_model() -> Result<nemotron::Nemotron> {
    let path = default_model_path();
    if !path.exists() {
        anyhow::bail!(
            "модель не найдена: {} (задайте STEALCODE_NEMOTRON_GGUF)",
            path.display()
        );
    }
    nemotron::Nemotron::load(&path)
}
