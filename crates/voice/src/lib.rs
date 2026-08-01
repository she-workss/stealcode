use std::{
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender},
    },
    time::Instant,
};

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::{
    models::whisper::{self as mwhisper, Config},
    quantized_var_builder::VarBuilder,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokenizers::Tokenizer;
use tracing::{error, info};

const WHISPER_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
];

#[derive(Debug, Clone)]
pub enum VoiceEvent {
    Status(String),
    Transcribed(String),
    Error(String),
}

#[derive(Debug, Clone)]
enum VoiceCommand {
    Start,
    Stop,
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

pub struct Decoder {
    model: mwhisper::quantized_model::Whisper,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    lang_token_ids: Vec<u32>,
}

impl Decoder {
    fn new(
        model: mwhisper::quantized_model::Whisper,
        tokenizer: Tokenizer,
        device: &Device,
    ) -> Result<Self> {
        let no_timestamps_token =
            token_id(&tokenizer, mwhisper::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, mwhisper::SOT_TOKEN)?;
        let transcribe_token =
            token_id(&tokenizer, mwhisper::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, mwhisper::EOT_TOKEN)?;
        let lang_token_ids: Vec<u32> = WHISPER_LANGUAGES
            .iter()
            .filter_map(|code| {
                token_id(&tokenizer, &format!("<|{code}|>")).ok()
            })
            .collect();
        if lang_token_ids.is_empty() {
            anyhow::bail!(
                "не найдено ни одного языкового токена в tokenizer.json"
            );
        }
        Ok(Self {
            model,
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
            no_timestamps_token,
            lang_token_ids,
        })
    }

    fn detect_language(
        &mut self,
        audio_features: &Tensor,
        device: &Device,
    ) -> Result<u32> {
        let probe = Tensor::new(&[self.sot_token], device)?.unsqueeze(0)?;
        let ys = self.model.decoder.forward(&probe, audio_features, true)?;
        let logits =
            self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let ids = Tensor::new(self.lang_token_ids.as_slice(), device)?;
        let lang_logits: Vec<f32> = logits.index_select(&ids, 0)?.to_vec1()?;
        let best = lang_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| self.lang_token_ids[i])
            .unwrap();
        Ok(best)
    }

    pub fn decode(&mut self, mel: &Tensor) -> Result<String> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0usize;
        let mut full_text = String::new();
        while seek < content_frames {
            let segment_size =
                usize::min(content_frames - seek, mwhisper::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_text = self.decode_segment(&mel_segment)?;
            if !segment_text.is_empty() {
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&segment_text);
            }
            seek += segment_size;
        }
        Ok(full_text.trim().to_string())
    }

    fn decode_segment(&mut self, mel: &Tensor) -> Result<String> {
        let device = mel.device().clone();
        let audio_features = self.model.encoder.forward(mel, true)?;
        let language_token = self.detect_language(&audio_features, &device)?;
        let sample_len = self.model.config.max_target_positions / 2;
        let mut tokens = vec![
            self.sot_token,
            language_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ];
        for i in 0..sample_len {
            let tokens_t =
                Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(
                &tokens_t,
                &audio_features,
                i == 0,
            )?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap();
            tokens.push(next_token);
            if next_token == self.eot_token
                || tokens.len() > self.model.config.max_target_positions
            {
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(text.trim().to_string())
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

fn download_file(url: &str, path: &std::path::Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("StealCode/0.1.0")
        .build()?;
    let mut resp = client.get(url).send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    let mut file = std::fs::File::create(path)?;
    std::io::copy(&mut resp, &mut file)?;
    Ok(())
}

pub struct LoadedModel {
    pub decoder: Decoder,
    pub config: Config,
    pub mel_filters: Vec<f32>,
    pub device: Device,
}

pub fn load_model(
    model_cache_dir: &std::path::Path,
    tx_event: &Sender<VoiceEvent>,
) -> Result<LoadedModel> {
    let model_id = "oxide-lab/whisper-base-GGUF".to_string();
    let weights_path = model_cache_dir.join("whisper-base-q8_0.gguf");
    if !weights_path.exists() {
        let _ = tx_event.send(VoiceEvent::Status(
            "Скачивание весов (~46 МБ)...".to_string(),
        ));
        let url = format!(
            "https://huggingface.co/{}/resolve/main/whisper-base-q8_0.gguf",
            model_id
        );
        download_file(&url, &weights_path)?;
    }
    let config_path = model_cache_dir.join("config.json");
    if !config_path.exists() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/config.json",
            model_id
        );
        download_file(&url, &config_path)?;
    }
    let tokenizer_path = model_cache_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            model_id
        );
        download_file(&url, &tokenizer_path)?;
    }
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {:?}", e))?;
    let mel_bytes: &[u8] = match config.num_mel_bins {
        80 => include_bytes!("../assets/melfilters.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    for (i, chunk) in mel_bytes.chunks(4).enumerate() {
        if chunk.len() == 4 {
            mel_filters[i] =
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    let device = Device::Cpu;
    let vb = VarBuilder::from_gguf(&weights_path, &device)?;
    let model = mwhisper::quantized_model::Whisper::load(&vb, config.clone())?;
    let decoder = Decoder::new(model, tokenizer, &device)?;
    Ok(LoadedModel {
        decoder,
        config,
        mel_filters,
        device,
    })
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
    let channels = stream_config.channels;
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut stream = None;
    let mut loaded_model: Option<LoadedModel> = None;
    let model_cache_dir = paths::data_dir().join("models").join("whisper");
    if let Err(e) = std::fs::create_dir_all(&model_cache_dir) {
        let _ = tx_event.send(VoiceEvent::Error(format!(
            "Не удалось создать папку: {:?}",
            e
        )));
        return;
    }
    while let Ok(cmd) = rx_cmd.recv() {
        match cmd {
            VoiceCommand::Start => {
                if loaded_model.is_none() {
                    let _ = tx_event.send(VoiceEvent::Status(
                        "Загрузка модели...".to_string(),
                    ));
                    match load_model(&model_cache_dir, &tx_event) {
                        Ok(m) => {
                            loaded_model = Some(m);
                        }
                        Err(e) => {
                            let _ = tx_event.send(VoiceEvent::Error(format!(
                                "Model load: {:?}",
                                e
                            )));
                            return;
                        }
                    }
                }
                let _ = tx_event
                    .send(VoiceEvent::Status("Идет запись...".to_string()));
                audio_buffer.lock().unwrap().clear();
                let buf_clone = Arc::clone(&audio_buffer);
                let stream_result = match supported_config.sample_format() {
                    cpal::SampleFormat::F32 => audio_device.build_input_stream(
                        stream_config.clone(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            buf_clone.lock().unwrap().extend_from_slice(data);
                        },
                        |e| error!("Audio stream error: {:?}", e),
                        None,
                    ),
                    _ => continue,
                };
                if let Ok(s) = stream_result {
                    if s.play().is_ok() {
                        stream = Some(s);
                    }
                }
            }
            VoiceCommand::Stop => {
                drop(stream.take());
                let raw_audio = audio_buffer.lock().unwrap().clone();
                if raw_audio.is_empty() {
                    continue;
                }
                if let Some(lm) = loaded_model.as_mut() {
                    let channels = channels as usize;
                    let mono: Vec<f32> = if channels <= 1 {
                        raw_audio
                    } else {
                        raw_audio
                            .chunks(channels)
                            .map(|c| c.iter().sum::<f32>() / channels as f32)
                            .collect()
                    };
                    let pcm_16k = if sample_rate != 16000 {
                        let ratio = 16000.0 / sample_rate as f32;
                        let out_len = (mono.len() as f32 * ratio) as usize;
                        let mut resampled = Vec::with_capacity(out_len);
                        for i in 0..out_len {
                            let src_idx = i as f32 / ratio;
                            let idx0 = src_idx.floor() as usize;
                            let idx1 = (idx0 + 1).min(mono.len() - 1);
                            let frac = src_idx - idx0 as f32;
                            resampled.push(
                                mono[idx0] * (1.0 - frac) + mono[idx1] * frac,
                            );
                        }
                        resampled
                    } else {
                        mono
                    };
                    let mel = mwhisper::audio::pcm_to_mel(
                        &lm.config,
                        &pcm_16k,
                        &lm.mel_filters,
                    );
                    let mel_len = mel.len();
                    let mel = match Tensor::from_vec(
                        mel,
                        (
                            1,
                            lm.config.num_mel_bins,
                            mel_len / lm.config.num_mel_bins,
                        ),
                        &lm.device,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = tx_event.send(VoiceEvent::Error(format!(
                                "Tensor error: {:?}",
                                e
                            )));
                            continue;
                        }
                    };
                    let t_decode = Instant::now();
                    match lm.decoder.decode(&mel) {
                        Ok(text) => {
                            info!(
                                "Decode voice message: {:?}",
                                t_decode.elapsed()
                            );
                            let _ =
                                tx_event.send(VoiceEvent::Transcribed(text));
                        }
                        Err(e) => {
                            let _ = tx_event.send(VoiceEvent::Error(format!(
                                "Decode error: {:?}",
                                e
                            )));
                        }
                    }
                }
                // Гарантированно возвращаем память ОС
                return;
            }
        }
    }
}
