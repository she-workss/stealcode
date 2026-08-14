//! Model-agnostic speech-to-text API.
//!
//! [`AsrModel`] describes a speech model; [`AsrModel::live`] starts a
//! streaming session behind the [`LiveAsr`] trait, so a service crate
//! (e.g. `voice`) works against these traits plus a [`LatencyMode`]
//! without knowing which model is underneath. The `Nemotron`
//! implementation lives in [`crate::nemotron`].

use anyhow::Result;

/// How much audio a streaming encoder step covers, trading latency
/// against throughput. The model's attention chunk is 56 mel frames
/// (560 ms): `Standard` is the smallest batch whose output is identical
/// to the offline encode; smaller batches cut the attention right
/// context at the batch end, trading lower latency for slightly less
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyMode {
    /// 80 ms (8 mel frames).
    UltraLow,
    /// 160 ms (16 mel frames).
    Low,
    /// 320 ms (32 mel frames).
    Medium,
    /// 560 ms (56 mel frames).
    Standard,
    /// 1120 ms (112 mel frames).
    HighQuality,
}

impl LatencyMode {
    /// Mel frames per streaming batch for this mode.
    #[must_use]
    pub const fn batch_mel(self) -> usize {
        match self {
            Self::UltraLow => 8,
            Self::Low => 16,
            Self::Medium => 32,
            Self::Standard => 56,
            Self::HighQuality => 112,
        }
    }
}

/// One decoded token with timing.
#[derive(Debug, Clone)]
pub struct SegmentToken {
    pub id: u32,
    pub p: f32,
    /// Encoder frame at emission; x `frame_to_ms` for milliseconds.
    pub step: usize,
}

/// A transcription result.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    /// Language tags stripped, in emit order (times relative).
    pub tokens: Vec<SegmentToken>,
}

/// A speech model that transcribes 16 kHz mono PCM.
pub trait AsrModel {
    /// Full offline transcription; `language` is a BCP-47 hint
    /// (None = auto).
    fn transcribe(
        &mut self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<Transcription>;

    /// Start a streaming session at the given latency mode. The model
    /// is passed into each [`LiveAsr`] call, so the session state is
    /// owned independently of the model borrow.
    fn live(&mut self, mode: LatencyMode) -> Result<Box<dyn LiveAsr<Self>>>;
}

/// A streaming transcription session.
pub trait LiveAsr<M: ?Sized> {
    /// Append 16 kHz mono PCM; commits text as batches complete.
    fn push(&mut self, model: &mut M, pcm: &[f32]) -> Result<()>;

    /// End of stream: encode and decode the leftover tail.
    fn flush(&mut self, model: &mut M) -> Result<()>;

    /// Committed transcript so far (language tags stripped).
    fn text(&self, model: &M) -> String;
}
