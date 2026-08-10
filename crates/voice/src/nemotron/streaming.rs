//! Incremental streaming encoder with per-frame caches.
//!
//! Instead of re-encoding a fixed left-context window on every step
//! (which recomputes the same left frames each batch), this encoder
//! caches the pre_encode output and every block's output per frame, so
//! a new batch only computes the frames it adds:
//!
//!   * pre_encode, FF1, FF2, final LN: only the new frames;
//!   * attention: new queries, keys/values over the band only;
//!   * conv module: new outputs, with its `conv_context_left`-frame causal
//!     context recomputed from cached pre-FF2 activations.
//!
//! The attention band is chunk-aligned exactly like the offline path
//! (`k_min = sat_sub(q/4 - left_chunks)*4`,
//! `k_max = min((q/4+1)*4, T)`), so the output for a frame is identical
//! to `Encoder::encode` of the same audio — live text is a strict
//! prefix of the offline transcript.
//!
//! Caches are sliding: frames older than the band/conv context are
//! dropped, so memory stays bounded for long sessions.

use std::time::Instant;

use anyhow::{Result, bail};
use rayon::prelude::*;

use super::{
    config::EncoderConfig,
    encoder::{Block, Encoder},
};

/// Incremental encoder state. All caches are time-major `[n, d_model]`
/// and start at absolute encoder frame `base` (sliding window).
#[derive(Debug)]
pub struct StreamingEncoder {
    /// pre_encode output cache.
    pre: Vec<f32>,
    /// Per-block output caches; `blocks[b]` feeds block `b+1`.
    blocks: Vec<Vec<f32>>,
    /// Per-block pre-FF2 activations (y after attention, before the
    /// conv module) — needed to rebuild the conv module's causal left
    /// context without recomputing whole blocks.
    pre_conv: Vec<Vec<f32>>,
    /// Per-block post-FF1 activations (y after macaron FF1, before the
    /// attention LN) — the attention's LayerNorm runs on these, so the
    /// band's old frames must be cached.
    ff1_y: Vec<Vec<f32>>,
    /// Per-block attention-LN outputs for the band frames
    /// [kv_lo, kv_hi). Computed once per frame; a new batch only adds
    /// its own new frames and drops the band frames that fell out of
    /// the left context.
    ln_kv: Vec<Vec<f32>>,
    /// Per-block K projections for the same band frames.
    k_v: Vec<Vec<f32>>,
    /// Per-block V projections for the same band frames.
    v_v: Vec<Vec<f32>>,
    /// Absolute first encoder frame of the band caches.
    kv_lo: usize,
    /// Per-block attn_pos projections for pos [-3..59] (computed once,
    /// time-major [pos][d]).
    pos_p: Vec<Vec<f32>>,
    /// Shared Q8 dequantization scratch.
    scratch: Vec<f32>,
    /// Reusable transpose scratch (see `transpose_into`).
    trans: Vec<f32>,
    /// Absolute encoder-frame index of the first element in every cache.
    base: usize,
    /// Total frames in the caches.
    total: usize,
    /// d_model (set on the first `encode_new`).
    d: usize,
}

impl StreamingEncoder {
    pub fn new(n_blocks: usize) -> Self {
        Self {
            pre: Vec::new(),
            blocks: vec![Vec::new(); n_blocks],
            pre_conv: vec![Vec::new(); n_blocks],
            ff1_y: vec![Vec::new(); n_blocks],
            ln_kv: vec![Vec::new(); n_blocks],
            k_v: vec![Vec::new(); n_blocks],
            v_v: vec![Vec::new(); n_blocks],
            kv_lo: 0,
            pos_p: Vec::new(),
            scratch: Vec::new(),
            trans: Vec::new(),
            base: 0,
            total: 0,
            d: 0,
        }
    }

    /// Number of encoder frames currently cached.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Encoder output frames `[from, to)` (absolute encoder-frame
    /// indices, must already be computed).
    pub fn frames(&self, from: usize, to: usize) -> Result<&[f32]> {
        if self.d == 0 {
            bail!("streaming encoder: nothing encoded yet");
        }
        if from < self.base || to > self.total || to <= from {
            bail!(
                "frames [{from}, {to}) outside cache [{}, {})",
                self.base,
                self.total
            );
        }
        let last = self
            .blocks
            .last()
            .ok_or_else(|| anyhow::anyhow!("streaming encoder: no blocks"))?;
        Ok(&last[(from - self.base) * self.d..(to - self.base) * self.d])
    }

    /// Append encoder frames for mel frames `[t0, t1)` (absolute mel
    /// indices into `mel`, time-major `[n, n_mels]`). `t0` must equal
    /// the current cached end; `t1 - t0` is either a chunk-aligned
    /// batch (BATCH_MEL) or the end-of-stream tail.
    pub fn encode_new(
        &mut self,
        enc: &mut Encoder,
        mel: &[f32],
        n_mels: usize,
        t0: usize,
        t1: usize,
        prompt_id: Option<u32>,
    ) -> Result<()> {
        let cfg = enc.cfg.clone();
        let d = cfg.d_model;
        let chunk = cfg.att_context_right + 1;
        let left_chunks = cfg.att_context_left / chunk;
        let num_prompts = cfg.num_prompts;
        let n_layers = cfg.n_layers;
        let conv_left = cfg.conv_context_left;
        let trim_margin = cfg.att_context_left.max(conv_left) + 16;
        if self.d == 0 {
            self.d = d;
        }
        debug_assert_eq!(self.d, d);

        let s = t0 / 8; // first new encoder frame (t0 is a multiple of 8)
        let e = s + (t1 - t0).div_ceil(8); // decoded end (exclusive)
        if s != self.total {
            bail!(
                "streaming encoder out of sync: expected frame {s}, \
                 cache has {}",
                self.total
            );
        }
        // Frames to compute: the chunk-aligned decoded end for regular
        // batches; for a non-chunk-aligned tail the offline t_enc of the
        // whole audio (`f(8s + tail) = s + f(tail)`, f = 3x t/2+1), so the
        // bands clip at exactly the same T as the offline encode.
        let e_extra = ((e - 1) / chunk + 1) * chunk;
        let t_enc_offline = |tail: usize| ((tail / 2 + 1) / 2 + 1) / 2 + 1;
        let t_new = if e_extra == e {
            e
        } else {
            s + t_enc_offline(t1 - t0)
        };
        if t_new <= s {
            return Ok(());
        }
        let c = t_new - s;

        // ---- pre_encode for mel [8s-24, 8*t_new) ----
        // A slice's conv stack (3x stride-2 k=3 pad(2,1)) needs its 2
        // leftmost final frames to be real, i.e. the slice must start
        // 2*8 mel frames before the first needed frame's receptive
        // field [8j-14, 8j+7); those 2 frames (rel 0,1) are discarded.
        let mel_lo = s * 8;
        let win_start = mel_lo.saturating_sub(24);
        let start_rel = (mel_lo - win_start) / 8;
        // The slice must not include zero padding past the real mel:
        // the conv stack's out-of-range skips then land on the same
        // absolute mel positions as the offline path. Shorten the
        // natural end only if it still covers the last needed frame's
        // taps (the conv0 taps of frame `j` reach mel 8j).
        let mel_len = mel.len() / n_mels;
        let rel_max = start_rel + c - 1;
        let t_mel = mel_len
            .saturating_sub(win_start)
            .min((8 * rel_max + 1).max(t_new * 8 - win_start));
        let mut win = vec![0.0f32; t_mel * n_mels];
        let have = (mel.len() / n_mels).saturating_sub(win_start);
        let take = have.min(t_mel);
        if take > 0 {
            win[..take * n_mels].copy_from_slice(
                &mel[win_start * n_mels..(win_start + take) * n_mels],
            );
        }
        let t_pre = Instant::now();
        let pe = enc.pre_encode_forward(&win, t_mel, n_mels);
        let t_pe = pe.len() / d;
        if t_pe < start_rel + c {
            bail!(
                "pre_encode produced too few frames: {t_pe} < {}",
                start_rel + c
            );
        }
        self.pre
            .extend_from_slice(&pe[start_rel * d..(start_rel + c) * d]);
        let pre = &self.pre;
        let dump_dir = std::env::var_os("STEALCODE_DUMP_DIR")
            .map(std::path::PathBuf::from);
        if let Some(dir) = &dump_dir {
            let bytes: Vec<u8> =
                pre.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("senc_pre.bin"), bytes).ok();
        }

        // ---- attention band ----
        let k_lo = (s / chunk).saturating_sub(left_chunks) * chunk;
        let k_hi = t_new;
        // pos = q - k: min is -3 (chunk start), max is 59 at the end of
        // every full chunk (q = 4m+3, k_min = (m-14)*4 -> pos = 59);
        // only the very first batches are smaller (k_min = 0).
        let pos_lo: isize = -3;

        // ---- pos projections (once) ----
        if self.pos_p.is_empty() {
            let n_pos = 63; // pos -3..=59
            let ln10000 = 10000.0f32.ln();
            let mut pet = vec![0.0f32; n_pos * d];
            for i in 0..n_pos {
                let pos = (i as isize - 3) as f32;
                let row = &mut pet[i * d..(i + 1) * d];
                for kk in 0..d / 2 {
                    let div = (-2.0 * (kk as f32) * ln10000 / d as f32).exp();
                    row[2 * kk] = (pos * div).sin();
                    row[2 * kk + 1] = (pos * div).cos();
                }
            }
            let pett = transpose(&pet, n_pos, d);
            for b in 0..n_layers {
                let mut p = Vec::new();
                enc.blocks[b].attn_pos.forward_t(
                    &mut self.scratch,
                    &pett,
                    n_pos,
                    &mut p,
                );
                self.pos_p.push(transpose(&p, d, n_pos));
            }
        }

        // ---- slide the band caches: drop frames that fell out of the
        // left context, then block_new computes only the new frames ----
        let kv_drop = (k_lo - self.kv_lo) * d;
        if kv_drop > 0 {
            for v in &mut self.ln_kv {
                v.drain(..kv_drop);
            }
            for v in &mut self.k_v {
                v.drain(..kv_drop);
            }
            for v in &mut self.v_v {
                v.drain(..kv_drop);
            }
        }
        self.kv_lo = k_lo;

        // ---- blocks ----
        let mut out_new: Vec<f32> = Vec::new();
        let mut acc: Vec<(&'static str, std::time::Duration)> = Vec::new();
        let t_blk0 = Instant::now();
        for b in 0..n_layers {
            let block = &enc.blocks[b];
            let input: &[f32] = if b == 0 { pre } else { &self.blocks[b - 1] };
            let pc = &self.pre_conv[b];
            let (no, nco, ny, ln_new, k_new, v_new) = block_new(
                &cfg,
                block,
                input,
                pc,
                &self.k_v[b],
                &self.v_v[b],
                &self.pos_p[b],
                self.kv_lo,
                self.base,
                s,
                t_new,
                k_lo,
                k_hi,
                pos_lo,
                &mut self.scratch,
                &mut self.trans,
                b == 0,
                &mut acc,
            )?;
            self.blocks[b].extend_from_slice(&no);
            self.pre_conv[b].extend_from_slice(&nco);
            self.ff1_y[b].extend_from_slice(&ny);
            self.ln_kv[b].extend_from_slice(&ln_new);
            self.k_v[b].extend_from_slice(&k_new);
            self.v_v[b].extend_from_slice(&v_new);
            out_new = no;
            if let Some(dir) = &dump_dir {
                let bytes: Vec<u8> = self.blocks[b]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                std::fs::write(dir.join(format!("senc_b{b}.bin")), bytes).ok();
            }
        }

        // ---- prompt MLP on the new frames (per-frame, like encode) ----
        let t_p0 = Instant::now();
        acc.push(("__blk_total", t_blk0.elapsed()));
        if let (Some(mlp), Some(pid)) = (&mut enc.prompt, prompt_id) {
            if (pid as usize) < num_prompts {
                let cat_in = d + num_prompts;
                let mut cat = vec![0.0f32; cat_in * c];
                for t in 0..c {
                    cat[t * cat_in..t * cat_in + d]
                        .copy_from_slice(&out_new[t * d..(t + 1) * d]);
                    cat[t * cat_in + d + pid as usize] = 1.0;
                }
                let xt = transpose(&cat, c, cat_in);
                let mut h = Vec::new();
                mlp.mlp0.forward_t(&mut self.scratch, &xt, c, &mut h);
                for v in &mut h {
                    *v = v.max(0.0);
                }
                let mut y = Vec::new();
                mlp.mlp2.forward_t(&mut self.scratch, &h, c, &mut y);
                let y = transpose(&y, d, c);
                let last = self.blocks.len() - 1;
                let n = self.blocks[last].len();
                self.blocks[last][n - c * d..].copy_from_slice(&y);
            }
        }

        // ---- trim old frames ----
        self.total = t_new;
        let keep = s.saturating_sub(trim_margin);
        if keep > self.base {
            let drop = (keep - self.base) * d;
            self.pre.drain(..drop);
            for v in &mut self.blocks {
                v.drain(..drop);
            }
            for v in &mut self.pre_conv {
                v.drain(..drop);
            }
            for v in &mut self.ff1_y {
                v.drain(..drop);
            }
            self.base = keep;
        }
        if std::env::var_os("STEALCODE_TIMING").is_some() {
            eprintln!(
                "[senc] t {s}->{t_new} (+{c}): pre {:.3}s blocks {:.3}s prompt {:.3}s",
                t_pre.elapsed().as_secs_f64(),
                t_blk0.elapsed().as_secs_f64(),
                t_p0.elapsed().as_secs_f64()
            );
            let mut sums: Vec<(&'static str, std::time::Duration)> = Vec::new();
            for (k, v) in &acc {
                match sums.iter_mut().find(|(sk, _)| *sk == *k) {
                    Some(e) => e.1 += *v,
                    None => sums.push((*k, *v)),
                }
            }
            let mut s = String::new();
            for (k, v) in &sums {
                s.push_str(&format!(" {k}={:.3}", v.as_secs_f64() * 1e3));
            }
            eprintln!("[senc] sec{}{s}", 0);
        }
        Ok(())
    }
}

/// Compute one conformer block's outputs for frames `[s, t_new)`
/// (absolute indices). `input` is the block's input cache (frames
/// `[base, t_new)`, old + new), `pre_conv_in` the same block's cached
/// pre-FF2 activations for the old frames, and `k_v_in`/`v_v_in` the
/// block's cached K/V projections for the band frames `[kv_lo, s)`.
/// Returns the new frames' block output, their pre-FF2 activations,
/// and the new band frames' LN/K/V.
#[allow(clippy::too_many_arguments)]
fn block_new(
    cfg: &EncoderConfig,
    b: &Block,
    input: &[f32],
    pre_conv_in: &[f32],
    k_v_in: &[f32],
    v_v_in: &[f32],
    pos_p: &[f32],
    kv_lo: usize,
    base: usize,
    s: usize,
    t_new: usize,
    k_lo: usize,
    k_hi: usize,
    pos_lo: isize,
    scratch: &mut Vec<f32>,
    trans: &mut Vec<f32>,
    dump0: bool,
    acc: &mut Vec<(&'static str, std::time::Duration)>,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let d = cfg.d_model;
    let n_heads = cfg.n_heads;
    let head_dim = d / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let chunk = cfg.att_context_right + 1;
    let left_chunks = cfg.att_context_left / chunk;
    let conv_left = cfg.conv_context_left;
    let c = t_new - s;
    let rel = |a: usize| a - base; // input-relative index
    let krel = |a: usize| a - kv_lo; // band-cache-relative index
    let conv_lo = s.saturating_sub(conv_left);
    let dmp0 = |name: &str, v: &[f32]| {
        if dump0 {
            if let Some(dir) = std::env::var_os("STEALCODE_DUMP_DIR") {
                let p = std::path::PathBuf::from(dir)
                    .join(format!("{name}_s{s}.bin"));
                let bytes: Vec<u8> =
                    v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(p, bytes).ok();
            }
        }
    };

    // ---- macaron FF1 over the new frames ----
    let mut y = vec![0.0f32; c * d];
    let t_a = Instant::now();
    for j in 0..c {
        b.norm_ff1.forward(
            &input[(rel(s) + j) * d..(rel(s) + j + 1) * d],
            &mut y[j * d..(j + 1) * d],
        );
    }
    acc.push(("ln_ff1", t_a.elapsed()));
    let xt = transpose_into(&y, c, d, trans);
    let mut h = Vec::new();
    let t0x = Instant::now();
    b.ff1_lin1.forward_t(scratch, xt, c, &mut h);
    acc.push(("ff1_lin1", t0x.elapsed()));
    for v in &mut h {
        *v = *v * (1.0 + (-*v).exp()).recip(); // silu
    }
    let mut f = Vec::new();
    let t0x = Instant::now();
    b.ff1_lin2.forward_t(scratch, &h, c, &mut f);
    acc.push(("ff1_lin2", t0x.elapsed()));
    let t_a = Instant::now();
    let f = transpose_into(&f, d, c, trans);
    for i in 0..c * d {
        y[i] = input[rel(s) * d + i] + 0.5 * f[i];
    }
    acc.push(("ff1_res", t_a.elapsed()));
    dmp0("senc_b0_ff1", &y);

    // ---- rel-pos MHSA over the band ----
    // The attention LayerNorm runs on the post-FF1 activations: old
    // band frames from the cached ff1 (ln_kv was built from it), new
    // frames from `y`. Only the new frames get LN + K/V projections;
    // the old band frames reuse the cached K/V.
    let ny = y.clone();
    let mut ln_new = vec![0.0f32; c * d];
    let t_a = Instant::now();
    for j in 0..c {
        b.norm_att
            .forward(&y[j * d..(j + 1) * d], &mut ln_new[j * d..(j + 1) * d]);
    }
    acc.push(("ln_att", t_a.elapsed()));
    let ln_new_t = transpose_into(&ln_new, c, d, trans);
    let mut k_new = Vec::new();
    let mut v_new = Vec::new();
    let t_a = Instant::now();
    b.attn_k.forward_t(scratch, ln_new_t, c, &mut k_new);
    b.attn_v.forward_t(scratch, ln_new_t, c, &mut v_new);
    let k_new = transpose(&k_new, d, c);
    let v_new = transpose(&v_new, d, c);
    let lnst = transpose_into(&ln_new, c, d, trans);
    let mut q_new = Vec::new();
    b.attn_q.forward_t(scratch, lnst, c, &mut q_new);
    let q_new = transpose(&q_new, d, c);
    acc.push(("qkv", t_a.elapsed()));
    dmp0("senc_b0_ln", &ln_new);

    // scores[q, kk, h] = scale * (q_u.k + q_v.p[pos]), pos = q - kk_abs;
    // outside the band no attention (equivalent to the -inf mask).
    // Old band frames use the cached K/V, new frames the local ones.
    let band = k_hi - k_lo;
    let mut scores = vec![0.0f32; c * band * n_heads];
    let t_a = Instant::now();
    scores
        .par_chunks_mut(band * n_heads)
        .enumerate()
        .for_each(|(qi, row)| {
            let qq = s + qi;
            let q_chunk = qq / chunk;
            let k_min = q_chunk.saturating_sub(left_chunks) * chunk;
            let k_max = ((q_chunk + 1) * chunk).min(k_hi);
            let (k0, k1) = (k_min - k_lo, k_max - k_lo);
            for h in 0..n_heads {
                let hd = h * head_dim;
                let uh = &b.pos_u[hd..hd + head_dim];
                let vh = &b.pos_v[hd..hd + head_dim];
                let qu = &q_new[qi * d + hd..qi * d + hd + head_dim];
                let qv = &q_new[qi * d + hd..qi * d + hd + head_dim];
                for kk in k0..k1 {
                    let fr = k_lo + kk;
                    let kk_d = if fr < s {
                        &k_v_in[krel(fr) * d + hd..krel(fr) * d + hd + head_dim]
                    } else {
                        &k_new[(fr - s) * d + hd..(fr - s) * d + hd + head_dim]
                    };
                    let pr =
                        (qq as isize - (k_lo + kk) as isize - pos_lo) as usize;
                    let pos_row = &pos_p[pr * d + hd..pr * d + hd + head_dim];
                    let mut acc = 0.0f32;
                    for i in 0..head_dim {
                        let qui = qu[i] + uh[i];
                        let qvi = qv[i] + vh[i];
                        acc += qui * kk_d[i] + qvi * pos_row[i];
                    }
                    row[kk * n_heads + h] = acc * scale;
                }
            }
        });
    dmp0("senc_b0_scores", &scores);
    acc.push(("scores", t_a.elapsed()));

    // softmax over the band + weighted sum of v. Each row (qi) works on
    // its own disjoint slices, so the in-place exp cache below stays
    // safe under rayon.
    let mut attn_out = vec![0.0f32; c * d];
    let t_a = Instant::now();
    scores
        .par_chunks_mut(band * n_heads)
        .zip(attn_out.par_chunks_mut(d))
        .enumerate()
        .for_each(|(qi, (srow, row))| {
            let qq = s + qi;
            let q_chunk = qq / chunk;
            let k_min = q_chunk.saturating_sub(left_chunks) * chunk;
            let k_max = ((q_chunk + 1) * chunk).min(k_hi);
            let (k0, k1) = (k_min - k_lo, k_max - k_lo);
            for h in 0..n_heads {
                let hd = h * head_dim;
                let mut maxv = f32::NEG_INFINITY;
                for kk in k0..k1 {
                    maxv = maxv.max(srow[kk * n_heads + h]);
                }
                let mut sum = 0.0f32;
                for kk in k0..k1 {
                    let e = (srow[kk * n_heads + h] - maxv).exp();
                    srow[kk * n_heads + h] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for dd in 0..head_dim {
                    let mut acc = 0.0f32;
                    for kk in k0..k1 {
                        let fr = k_lo + kk;
                        let vv = if fr < s {
                            &v_v_in[krel(fr) * d + hd + dd]
                        } else {
                            &v_new[(fr - s) * d + hd + dd]
                        };
                        acc += srow[kk * n_heads + h] * inv * vv;
                    }
                    row[hd + dd] = acc;
                }
            }
        });
    acc.push(("softmax", t_a.elapsed()));
    let at = transpose_into(&attn_out, c, d, trans);
    let mut o = Vec::new();
    let t_a = Instant::now();
    b.attn_out.forward_t(scratch, at, c, &mut o);
    let o = transpose_into(&o, d, c, trans);
    for i in 0..c * d {
        y[i] += o[i];
    }
    acc.push(("attn_out", t_a.elapsed()));
    dmp0("senc_b0_attn_res", &y);

    // ---- save the new frames' pre-FF2 activations ----
    let nco = y.clone();

    // ---- conv module ----
    // LN_conv + pw1 over [conv_lo, t_new): old frames from the cache,
    // new frames from y. dw output frame ot == abs frame conv_lo + ot
    // (for the first batch conv_lo == s, the dw conv supplies the left
    // pad itself), so abs frames [s, t_new) start at rel old_glu.
    let n_glu = t_new - conv_lo;
    let old_glu = s - conv_lo; // [conv_lo, s) from the cache
    let t_a = Instant::now();
    let mut lnc = vec![0.0f32; n_glu * d];
    for j in 0..old_glu {
        b.norm_conv.forward(
            &pre_conv_in[(rel(conv_lo) + j) * d..(rel(conv_lo) + j + 1) * d],
            &mut lnc[j * d..(j + 1) * d],
        );
    }
    for j in 0..c {
        b.norm_conv.forward(
            &y[j * d..(j + 1) * d],
            &mut lnc[(old_glu + j) * d..(old_glu + j + 1) * d],
        );
    }
    let lt = transpose_into(&lnc, n_glu, d, trans);
    let mut h2 = Vec::new();
    b.pw1.forward_t(scratch, lt, n_glu, &mut h2);
    let h2 = transpose_into(&h2, 2 * d, n_glu, trans);
    let mut glu = vec![0.0f32; n_glu * d];
    for tt in 0..n_glu {
        for i in 0..d {
            let gate = h2[tt * 2 * d + i];
            let val = h2[tt * 2 * d + d + i];
            glu[tt * d + i] = gate * (1.0 + (-val).exp()).recip();
        }
    }
    let mut conv = Vec::new();
    b.dw.forward(&glu, n_glu, &mut conv);
    dmp0("senc_b0_glu", &glu);
    dmp0("senc_b0_dw", &conv);
    let mut ln2 = vec![0.0f32; d];
    let mut o2 = vec![0.0f32; c * d];
    for j in 0..c {
        b.conv_ln
            .forward(&conv[(old_glu + j) * d..(old_glu + j + 1) * d], &mut ln2);
        for i in 0..d {
            o2[j * d + i] = ln2[i] * (1.0 + (-ln2[i]).exp()).recip(); // silu
        }
    }
    let ot = transpose_into(&o2, c, d, trans);
    let mut o2t = Vec::new();
    b.pw2.forward_t(scratch, ot, c, &mut o2t);
    let o2 = transpose_into(&o2t, d, c, trans);
    for i in 0..c * d {
        y[i] += o2[i];
    }
    acc.push(("conv", t_a.elapsed()));

    // ---- macaron FF2 over the new frames ----
    let mut y2 = vec![0.0f32; c * d];
    let t_a = Instant::now();
    for j in 0..c {
        b.norm_ff2
            .forward(&y[j * d..(j + 1) * d], &mut y2[j * d..(j + 1) * d]);
    }
    acc.push(("ln_ff2", t_a.elapsed()));
    let xt2 = transpose_into(&y2, c, d, trans);
    let mut h3 = Vec::new();
    let t_a = Instant::now();
    b.ff2_lin1.forward_t(scratch, xt2, c, &mut h3);
    acc.push(("ff2_lin1", t_a.elapsed()));
    for v in &mut h3 {
        *v = *v * (1.0 + (-*v).exp()).recip(); // silu
    }
    let mut f3 = Vec::new();
    let t_a = Instant::now();
    b.ff2_lin2.forward_t(scratch, &h3, c, &mut f3);
    acc.push(("ff2_lin2", t_a.elapsed()));
    let f3 = transpose_into(&f3, d, c, trans);
    for i in 0..c * d {
        y2[i] = y[i] + 0.5 * f3[i];
    }

    // ---- final per-block LN ----
    let mut out = vec![0.0f32; c * d];
    let t_a = Instant::now();
    for j in 0..c {
        b.norm_out
            .forward(&y2[j * d..(j + 1) * d], &mut out[j * d..(j + 1) * d]);
    }
    acc.push(("ln_out", t_a.elapsed()));
    Ok((out, nco, ny, ln_new, k_new, v_new))
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

/// Same as [`transpose`] but into a reusable scratch buffer.
fn transpose_into<'a>(
    x: &[f32],
    rows: usize,
    cols: usize,
    trans: &'a mut Vec<f32>,
) -> &'a mut Vec<f32> {
    trans.clear();
    trans.reserve(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            trans.push(x[r * cols + c]);
        }
    }
    trans
}
