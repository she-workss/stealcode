//! Optional GPU backend for the nemotron encoder (feature `voice/gpu`).
//!
//! Everything in this module is compiled only when the `gpu` feature is
//! enabled and is never required at runtime: if `GpuContext::init()`
//! returns `None`, the voice crate keeps using the CPU path untouched.

pub mod batch;
pub mod context;
pub mod encoder;
pub mod kernels;
pub mod model;
pub mod streaming;

use anyhow::Result;
pub use context::GpuContext;

use crate::{
    model::{LatencyMode, Transcription},
    nemotron::{
        Nemotron,
        live::{LiveTranscriber, StreamState},
    },
};

/// Full transcription of a WAV-sized utterance using the GPU for the
/// conformer blocks. Returns `Ok(None)` when no GPU is available (caller
/// falls back to the CPU path); `Err` is reserved for the case where a
/// GPU exists but initialization or inference fails.
///
/// Uses the GPU *streaming* encoder (one gpu submit per batch, as the
/// live path), then decodes the whole utterance in one shot, so the
/// result matches `Nemotron::transcribe` up to float rounding. `mode`
/// controls the encoder batch size (see [`LatencyMode`]).
pub fn transcribe_wav(
    model: &mut Nemotron,
    pcm: &[f32],
    language: Option<&str>,
    mode: LatencyMode,
) -> Result<Option<Transcription>> {
    let Some(senc) = streaming::try_build(&model.encoder)? else {
        return Ok(None);
    };
    let prompt_id = model.resolve_prompt_id(language)?;
    let st = StreamState::new(&model.decoder);
    let mut tr = LiveTranscriber::with_encoder(
        prompt_id,
        st,
        senc,
        mode.batch_mel(),
    );
    tr.push(model, pcm)?;
    tr.flush(model)?;
    let text = tr.text(model);
    let tokens = tr
        .tokens()
        .iter()
        .filter(|t| !model.tokenizer.is_strippable(t.id))
        .map(|t| crate::model::SegmentToken {
            id: t.id,
            p: t.p,
            step: t.step,
        })
        .collect();
    Ok(Some(Transcription { text, tokens }))
}
