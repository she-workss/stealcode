//! Streaming (incremental) RNN-T transcription.
//!
//! A [`LiveTranscriber`] encodes only the newly arrived audio on each
//! `push` (via the cached [`StreamingEncoder`]), decodes the new frames
//! with a carried RNN-T predictor state and appends the new text to the
//! committed transcript. The live text is a strict prefix of the offline
//! `Nemotron::transcribe` result — later audio never rewrites earlier
//! text. State is kept separate from the `&mut Nemotron`, so a caller
//! (e.g. the voice worker) can own the model and the transcriber side by
//! side without a self-referential borrow.

use anyhow::{Context, Result, bail};

use super::{
    Nemotron, Token, decoder::GreedyDecoder, streaming::StreamingEncoder,
};

/// One chunk = att_right+1 = 4 encoder frames = 32 mel frames = 320 ms.
pub const CHUNK_MEL: usize = 32;
/// Process this many 320 ms chunks per batch (the encoder caches
/// per-frame state, so batching chunks costs about the same in total as
/// one big batch while text streams with bounded latency).
pub const BATCH_CHUNKS: usize = 2;
pub const BATCH_MEL: usize = CHUNK_MEL * BATCH_CHUNKS;

/// Carried RNN-T predictor state. Tokens emitted so far are committed;
/// the state is a deterministic function of the token sequence, so
/// continuing it across chunks is exact (no re-decoding).
#[derive(Debug)]
pub struct StreamState {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    nh: Vec<Vec<f32>>,
    nc: Vec<Vec<f32>>,
    embed_x: Vec<f32>,
    last_token: i32,
}

impl StreamState {
    pub fn new(decoder: &GreedyDecoder) -> Self {
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
pub fn decode_frames(
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
pub fn grow_mel(
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

/// Incremental transcript state (encoder cache + predictor state +
/// accumulated tokens), decoupled from the `Nemotron` so the voice
/// worker can own both independently.
#[derive(Debug)]
pub struct LiveTranscriber {
    prompt_id: u32,
    pcm: Vec<f32>,
    mel: Vec<f32>,
    mel_next: usize,
    processed_mel: usize,
    tokens: Vec<Token>,
    st: StreamState,
    scratch: Vec<f32>,
    senc: StreamingEncoder,
}

impl LiveTranscriber {
    pub fn new(model: &Nemotron, prompt_id: u32) -> Self {
        let st = StreamState::new(&model.decoder);
        let senc = StreamingEncoder::new(model.encoder.cfg.n_layers);
        Self {
            prompt_id,
            pcm: Vec::new(),
            mel: Vec::new(),
            mel_next: 0,
            processed_mel: 0,
            tokens: Vec::new(),
            st,
            scratch: Vec::new(),
            senc,
        }
    }

    /// Append 16 kHz mono PCM; grows mel and processes complete batches.
    pub fn push(
        &mut self,
        model: &mut Nemotron,
        samples: &[f32],
    ) -> Result<()> {
        self.pcm.extend_from_slice(samples);
        grow_mel(model, &self.pcm, &mut self.mel, &mut self.mel_next, false);
        while self.mel_next - self.processed_mel >= BATCH_MEL {
            self.process(model, BATCH_MEL)?;
        }
        Ok(())
    }

    /// End of stream: encode+decode the leftover tail (< BATCH_MEL).
    pub fn flush(&mut self, model: &mut Nemotron) -> Result<()> {
        grow_mel(model, &self.pcm, &mut self.mel, &mut self.mel_next, true);
        let total = model.frontend.n_frames(self.pcm.len()).min(self.mel_next);
        let tail = total.saturating_sub(self.processed_mel);
        if tail > 0 {
            self.process(model, tail)?;
        }
        Ok(())
    }

    /// Committed tokens so far.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Mel frames computed so far.
    pub fn mel_next(&self) -> usize {
        self.mel_next
    }

    /// Current committed transcript (language tags stripped).
    pub fn text(&self, model: &Nemotron) -> String {
        let ids: Vec<u32> = self.tokens.iter().map(|t| t.id).collect();
        model.tokenizer.decode_transcript(&ids, true)
    }

    /// Encode+decode `new_mel` frames of new audio. The streaming
    /// encoder caches every frame, so the encoder output of the new
    /// frames is identical to an offline full-audio encode, and the
    /// committed prefix is stable.
    fn process(&mut self, model: &mut Nemotron, new_mel: usize) -> Result<()> {
        let n_mels = model.cfg.preprocessor.n_mels;
        let t0_mel = self.processed_mel;
        self.senc.encode_new(
            &mut model.encoder,
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
            model,
            frames,
            count,
            &mut self.st,
            &mut self.tokens,
            &mut self.scratch,
        )
        .context("live rnnt decode")?;
        self.processed_mel += new_mel;
        Ok(())
    }
}

/// Transpose [rows, cols] time-major -> [cols, rows] row-major.
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            out.push(x[r * cols + c]);
        }
    }
    out
}
