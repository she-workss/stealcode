//! nemotron-3.5-asr-streaming-0.6b RNNT-ASR in pure Rust.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod live;
pub(crate) mod timing;
pub mod weights;

use std::path::Path;

use anyhow::{Context, Result};
pub use config::ModelConfig;
pub use decoder::{GreedyDecoder, Token};
pub use encoder::Encoder;
use tracing::debug;

use crate::math::f32_bytes;
pub use crate::{
    dsp::MelFrontend,
    model::{AsrModel, LatencyMode, LiveAsr, SegmentToken, Transcription},
    tokenizer::Tokenizer,
};

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
        let gguf = crate::gguf::Gguf::open(path)?;
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
        if let Some(dir) = timing::dump_dir() {
            let bytes = f32_bytes(&mel);
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

impl AsrModel for Nemotron {
    fn transcribe(
        &mut self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<Transcription> {
        Nemotron::transcribe(self, pcm, language)
    }

    fn live(&mut self, mode: LatencyMode) -> Result<Box<dyn LiveAsr<Self>>> {
        let prompt_id = self.resolve_prompt_id(None)?;
        Ok(Box::new(live::LiveTranscriber::build(
            self, prompt_id, mode,
        )?))
    }
}
