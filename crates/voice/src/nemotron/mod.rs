//! nemotron-3.5-asr-streaming-0.6b RNNT-ASR in pure Rust.
//!
//! Port of the whisper.cpp-family C++ reference in
//! `C:\Users\sinke\AppData\Local\Temp\opencode\tcpp` (numeric parity
//! target). Offline path: mel -> FastConformer encoder (+ prompt MLP)
//! -> RNNT greedy decode -> SentencePiece transcript.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod gguf;
pub mod live;
pub mod preprocess;
pub mod sgemm_kernel;
pub mod streaming;
pub mod tokenizer;
pub mod weights;

use std::path::Path;

use anyhow::{Context, Result};
pub use config::ModelConfig;
pub use decoder::{GreedyDecoder, Token};
pub use encoder::Encoder;
pub use preprocess::MelFrontend;
pub use tokenizer::Tokenizer;
use tracing::debug;

/// One decoded token with timing (matches the C++ TokenEntry).
#[derive(Debug, Clone)]
pub struct SegmentToken {
    pub id: u32,
    pub p: f32,
    /// Encoder frame at emission; x frame_to_ms for milliseconds.
    pub step: usize,
}

#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    /// Stripped of language tags, in emit order (times relative).
    pub tokens: Vec<SegmentToken>,
}

#[derive(Debug)]
pub struct Nemotron {
    pub cfg: ModelConfig,
    pub frontend: MelFrontend,
    pub encoder: Encoder,
    pub decoder: GreedyDecoder,
    pub tokenizer: Tokenizer,
    /// 80 ms per encoder frame (subsampling 8 * hop 160 / 16 kHz).
    pub frame_to_ms: f64,
}

impl Nemotron {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = gguf::Gguf::open(path)?;
        let cfg = ModelConfig::from_gguf(&gguf)?;

        let fb = gguf
            .tensor("preprocessor.fb")
            .context("GGUF tensor preprocessor.fb not found")?;
        let fb = gguf.read_f32(fb)?;
        let frontend = MelFrontend::new(cfg.preprocessor.clone(), &fb)?;

        let encoder = Encoder::load(&gguf, cfg.encoder.clone())?;
        let decoder = GreedyDecoder::load(&gguf, &cfg.rnnt)?;
        let tokenizer = Tokenizer::new(&cfg);

        let frame_to_ms = 1000.0
            * cfg.encoder.subsampling_factor as f64
            * cfg.preprocessor.hop as f64
            / cfg.preprocessor.sample_rate as f64;

        Ok(Self {
            cfg,
            frontend,
            encoder,
            decoder,
            tokenizer,
            frame_to_ms,
        })
    }

    /// Resolve a BCP-47 language hint to a prompt index (None -> the
    /// auto slot, typically en-US).
    pub fn resolve_prompt_id(&self, language: Option<&str>) -> Result<u32> {
        self.cfg.resolve_prompt_id(language)
    }

    /// Full offline transcription of 16 kHz mono PCM.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<Transcription> {
        let prompt_id = self.resolve_prompt_id(language)?;
        let mel = self.frontend.compute(pcm)?;
        let t_mel = mel.len() / self.cfg.preprocessor.n_mels;
        let mut enc = Vec::new();
        let t_enc =
            self.encoder
                .encode(&mel, t_mel, Some(prompt_id), &mut enc)?;
        let raw = self.decoder.decode(&enc, t_enc)?;

        let ids: Vec<u32> = raw.iter().map(|t| t.id).collect();
        let strip_tags = true;
        let text = self.tokenizer.decode_transcript(&ids, strip_tags);
        let tokens = raw
            .into_iter()
            .filter(|t| !self.tokenizer.is_strippable(t.id))
            .map(|t| SegmentToken {
                id: t.id,
                p: t.p,
                step: t.step,
            })
            .collect();
        Ok(Transcription { text, tokens })
    }

    /// Raw token stream (language tags included, whitespace
    /// unnormalized), for parity debugging.
    pub fn transcribe_raw(
        &mut self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<Vec<Token>> {
        let prompt_id = self.resolve_prompt_id(language)?;
        let mel = self.frontend.compute(pcm)?;
        let t_mel = mel.len() / self.cfg.preprocessor.n_mels;
        debug!(
            "mel: {} frames, first: {:?} last: {:?}",
            t_mel,
            &mel[..4],
            &mel[mel.len() - 4..]
        );
        if let Some(dir) =
            std::env::var_os("STEALCODE_DUMP_DIR").map(std::path::PathBuf::from)
        {
            let bytes: Vec<u8> =
                mel.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("mel.bin"), bytes).ok();
        }
        let mut enc = Vec::new();
        let t_enc =
            self.encoder
                .encode(&mel, t_mel, Some(prompt_id), &mut enc)?;
        let nan = enc.iter().filter(|v| v.is_nan()).count();
        let inf = enc.iter().filter(|v| v.is_infinite()).count();
        let mx = enc.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        debug!(
            "enc: t={t_enc} nan={nan} inf={inf} maxabs={mx} first={:?}",
            &enc[..4]
        );
        self.decoder.decode(&enc, t_enc)
    }
}
