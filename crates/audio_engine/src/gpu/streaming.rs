//! GPU incremental streaming encoder.
//!
//! Mirrors `crate::nemotron::streaming::StreamingEncoder`: the same
//! per-frame caches (`pre`, per-block outputs, pre-FF2 activations, band
//! K/V) and the same chunk-aligned attention band, so the output for any
//! frame equals the offline GPU encode. The heavy per-block work (GEMMs,
//! LayerNorms, streaming attention, dw conv) runs on the GPU: all of a
//! batch's blocks are recorded into one [`ComputeBatch`] and submitted
//! once (a block's GPU output feeds the next block in-device), then the
//! persistent per-block results are pulled back with a single batched
//! download - one submit + two device polls per batch, instead of one
//! submit per block. pre_encode and the tiny prompt MLP stay on the CPU
//! (convs + 2-layer MLP are small). Caches are host-side for now -
//! moving them to persistent GPU buffers is a later optimization.

use std::sync::Arc;

use anyhow::{Result, bail};

use super::{
    batch::ComputeBatch,
    context::GpuContext,
    kernels::{
        AttnStreamKernel, DwConvKernel, ElementwiseKernel, LayerNormKernel,
        Q8Gemm, bytemuck_safe, bytes_to_f32,
    },
    model::{GpuBlock, GpuModel},
};
use crate::{math::transpose, nemotron::encoder::Encoder};

/// Kernel handles shared by every block.
#[derive(Debug)]
pub struct BlockKernels {
    pub gemm: Q8Gemm,
    pub ln: LayerNormKernel,
    pub ew: ElementwiseKernel,
    pub attn: AttnStreamKernel,
    pub dw: DwConvKernel,
}

impl BlockKernels {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        Ok(Self {
            gemm: Q8Gemm::new(ctx)?,
            ln: LayerNormKernel::new(ctx)?,
            ew: ElementwiseKernel::new(ctx)?,
            attn: AttnStreamKernel::new(ctx)?,
            dw: DwConvKernel::new(ctx)?,
        })
    }
}

/// Incremental GPU encoder state. All caches are time-major `[n, d_model]`
/// and start at absolute encoder frame `base` (sliding window).
#[derive(Debug)]
pub struct GpuStreamingEncoder {
    kern: BlockKernels,
    model: GpuModel,
    /// pre_encode output cache.
    pre: Vec<f32>,
    /// Per-block output caches; `blocks[b]` feeds block `b+1`.
    blocks: Vec<Vec<f32>>,
    /// Per-block GLU outputs (post LN_conv + pw1) - the dw conv's
    /// causal left context, computed once per frame instead of being
    /// rebuilt from the pre-FF2 activations on every batch (see the CPU
    /// encoder for the rationale).
    glu: Vec<Vec<f32>>,
    /// Per-block K projections for band frames `[kv_lo, s)`.
    k_v: Vec<Vec<f32>>,
    /// Per-block V projections for the same band frames.
    v_v: Vec<Vec<f32>>,
    /// Absolute first encoder frame of the band caches.
    kv_lo: usize,
    /// Per-block attn_pos projections for pos [-3..59] (computed once,
    /// time-major `[pos][d]`).
    pos_p: Vec<Vec<f32>>,
    /// Reusable CPU scratch (prompt MLP).
    scratch: Vec<f32>,
    /// Absolute encoder-frame index of the first element in every cache.
    base: usize,
    /// Total frames in the caches.
    total: usize,
    /// d_model (set on the first `encode_new`).
    d: usize,
}

impl GpuStreamingEncoder {
    pub fn new(ctx: &Arc<GpuContext>, model: GpuModel) -> Result<Self> {
        let n_blocks = model.blocks.len();
        Ok(Self {
            kern: BlockKernels::new(ctx)?,
            model,
            pre: Vec::new(),
            blocks: vec![Vec::new(); n_blocks],
            glu: vec![Vec::new(); n_blocks],
            k_v: vec![Vec::new(); n_blocks],
            v_v: vec![Vec::new(); n_blocks],
            kv_lo: 0,
            pos_p: Vec::new(),
            scratch: Vec::new(),
            base: 0,
            total: 0,
            d: 0,
        })
    }

    /// Number of encoder frames currently cached.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Encoder output frames `[from, to)` (absolute encoder-frame
    /// indices, must already be computed).
    pub fn frames(&self, from: usize, to: usize) -> Result<&[f32]> {
        if self.d == 0 {
            bail!("gpu streaming encoder: nothing encoded yet");
        }
        if from < self.base || to > self.total || to <= from {
            bail!(
                "gpu frames [{from}, {to}) outside cache [{}, {})",
                self.base,
                self.total
            );
        }
        let last = self.blocks.last().ok_or_else(|| {
            anyhow::anyhow!("gpu streaming encoder: no blocks")
        })?;
        Ok(&last[(from - self.base) * self.d..(to - self.base) * self.d])
    }

    /// Append encoder frames for mel frames `[t0, t1)`, exactly like
    /// `StreamingEncoder::encode_new` but computing each block on the GPU.
    pub fn encode_new(
        &mut self,
        enc: &mut Encoder,
        mel: &[f32],
        n_mels: usize,
        t0: usize,
        t1: usize,
        prompt_id: Option<u32>,
        fin: bool,
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

        let s = t0 / 8; // first new encoder frame (t0 is a multiple of 8)
        let e = s + (t1 - t0).div_ceil(8); // decoded end (exclusive)
        if s != self.total {
            bail!(
                "gpu streaming encoder out of sync: expected frame {s}, \
                 cache has {}",
                self.total
            );
        }
        let t_enc_offline = |tail: usize| ((tail / 2 + 1) / 2 + 1) / 2 + 1;
        // Same contract as the CPU encoder: intermediate batches end the
        // cache exactly at their decoded end (`e`), so the next call
        // stays in sync for any batch size; the final tail (`fin`)
        // computes all `f(tail)` frames and the decoder consumes them,
        // so the transcript ends where the offline encode does.
        let t_new = if fin { s + t_enc_offline(t1 - t0) } else { e };
        if t_new <= s {
            return Ok(());
        }
        let c = t_new - s;

        // ---- pre_encode for mel [8s-24, 8*t_new) (CPU convs) ----
        let mel_lo = s * 8;
        let win_start = mel_lo.saturating_sub(24);
        let start_rel = (mel_lo - win_start) / 8;
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
        let pe = enc.pre_encode_forward(&win, t_mel, n_mels);
        let t_pe = pe.len() / d;
        if t_pe < start_rel + c {
            bail!(
                "gpu pre_encode produced too few frames: {t_pe} < {}",
                start_rel + c
            );
        }
        self.pre
            .extend_from_slice(&pe[start_rel * d..(start_rel + c) * d]);

        // ---- attention band ----
        let k_lo = (s / chunk).saturating_sub(left_chunks) * chunk;
        let k_hi = t_new;

        // ---- pos projections (once) ----
        if self.pos_p.is_empty() {
            let pos_lo: isize = 1 - chunk as isize;
            let n_pos = (left_chunks + 2) * chunk - 1;
            let pos_lo_cast = pos_lo as f32;
            let ln10000 = 10000.0f32.ln();
            let mut pet = vec![0.0f32; n_pos * d];
            for i in 0..n_pos {
                let pos = i as f32 + pos_lo_cast;
                let row = &mut pet[i * d..(i + 1) * d];
                for kk in 0..d / 2 {
                    let div = (-2.0 * (kk as f32) * ln10000 / d as f32).exp();
                    row[2 * kk] = (pos * div).sin();
                    row[2 * kk + 1] = (pos * div).cos();
                }
            }
            for b in 0..n_layers {
                let lin = &self.model.blocks[b].attn_pos;
                let p = self.kern.gemm.gemm(
                    &lin.packed,
                    &lin.q,
                    &lin.s,
                    lin.bias.as_ref(),
                    &pet,
                    n_pos,
                    lin.k,
                );
                self.pos_p.push(p);
            }
        }

        // ---- slide the band caches ----
        let kv_drop = (k_lo - self.kv_lo) * d;
        if kv_drop > 0 {
            for v in &mut self.k_v {
                v.drain(..kv_drop);
            }
            for v in &mut self.v_v {
                v.drain(..kv_drop);
            }
        }
        self.kv_lo = k_lo;

        // ---- blocks ----
        // All blocks are recorded into one submission; the previous
        // block's GPU output feeds the next block's input in-device
        // (`input_gpu`), and all persistent results are downloaded with
        // a single batched readback after the submit.
        let st = std::env::var_os("STEALCODE_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let t_rec = t0;
        let mut batch = ComputeBatch::new(&self.model.ctx);
        let act = (c * d * 4) as u64;
        let mut downloads: Vec<(wgpu::Buffer, u64)> =
            Vec::with_capacity(4 * n_layers);
        let mut prev_out: Option<wgpu::Buffer> = None;
        for b in 0..n_layers {
            let input: &[f32] = if b == 0 {
                &self.pre
            } else {
                &self.blocks[b - 1]
            };
            let (out, glu_new, k_new, v_new) = block_new(
                &mut self.kern,
                &mut batch,
                &self.model,
                &self.model.blocks[b],
                d,
                input,
                prev_out.as_ref(),
                &self.glu[b],
                &self.k_v[b],
                &self.v_v[b],
                &self.pos_p[b],
                self.base,
                s,
                t_new,
                k_lo,
                k_hi,
                chunk,
                left_chunks,
                conv_left,
            );
            downloads.extend([
                (out.clone(), act),
                (glu_new, act),
                (k_new, act),
                (v_new, act),
            ]);
            prev_out = Some(out);
        }
        let t_rec_ms = t_rec.elapsed().as_secs_f64() * 1e3;
        let t_sub = std::time::Instant::now();
        batch.submit();
        let t_sub_ms = t_sub.elapsed().as_secs_f64() * 1e3;
        let t_dl0 = std::time::Instant::now();
        let mut dl = self.model.ctx.download_many(&downloads).into_iter();
        let t_dl_ms = t_dl0.elapsed().as_secs_f64() * 1e3;
        let t_host = std::time::Instant::now();
        let mut out_new: Vec<f32> = Vec::new();
        for b in 0..n_layers {
            let no = bytes_to_f32(&dl.next().expect("out download"), c * d);
            let glu_new =
                bytes_to_f32(&dl.next().expect("glu download"), c * d);
            let k_new = bytes_to_f32(&dl.next().expect("k download"), c * d);
            let v_new = bytes_to_f32(&dl.next().expect("v download"), c * d);
            self.blocks[b].extend_from_slice(&no);
            self.glu[b].extend_from_slice(&glu_new);
            self.k_v[b].extend_from_slice(&k_new);
            self.v_v[b].extend_from_slice(&v_new);
            out_new = no;
        }
        let t_host_ms = t_host.elapsed().as_secs_f64() * 1e3;
        if st {
            eprintln!(
                "[senc] gpu {s}->{t_new} (+{c}): rec {t_rec_ms:.1}ms submit {t_sub_ms:.1}ms dl {t_dl_ms:.1}ms host {t_host_ms:.1}ms total {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }

        // ---- prompt MLP on the new frames (CPU, like encode) ----
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
            for v in &mut self.glu {
                v.drain(..drop);
            }
            self.base = keep;
        }
        Ok(())
    }
}

/// Compute one conformer block's outputs for frames `[s, t_new)`
/// (absolute indices) on the GPU. `input` is the block's input cache
/// (frames `[base, t_new)`, old + new), `input_gpu` the previous block's
/// GPU output buffer (the new frames of `input` - `Some` inside a batch,
/// `None` for the first block, which uploads from `input` instead),
/// `glu_in` the block's cached GLU outputs for frames `[conv_lo, s)`
/// (the dw conv's left context), `k_v_in`/`v_v_in` the block's cached
/// K/V for band frames `[kv_lo, s)`. All dispatches are recorded into
/// `batch` (no submit); the caller submits once after all blocks and
/// downloads the persistent results. Returns the GPU buffers for `out`,
/// `glu_new`, `k_new`, `v_new` (each `c * d` frames).
#[allow(clippy::too_many_arguments)]
fn block_new(
    kern: &mut BlockKernels,
    batch: &mut ComputeBatch<'_>,
    m: &GpuModel,
    b: &GpuBlock,
    d: usize,
    input: &[f32],
    input_gpu: Option<&wgpu::Buffer>,
    glu_in: &[f32],
    k_v_in: &[f32],
    v_v_in: &[f32],
    pos_p: &[f32],
    base: usize,
    s: usize,
    t_new: usize,
    k_lo: usize,
    k_hi: usize,
    chunk: usize,
    left_chunks: usize,
    conv_left: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let scale = 1.0 / (m.head_dim as f32).sqrt();
    let c = t_new - s;
    let rel = |a: usize| a - base; // input-relative index
    let conv_lo = s.saturating_sub(conv_left);
    let old_glu = s - conv_lo; // [conv_lo, s) from the cache
    let n_glu = t_new - conv_lo;
    let act = (c * d * 4) as u64;
    let glu_act = (n_glu * d * 4) as u64;
    let dff = b.ff1_lin1.packed.rows;
    let dff_size = (c * dff * 4) as u64;
    let pw1_size = (c * b.pw1.packed.rows * 4) as u64;
    let band = k_hi - k_lo;
    let band_size = (band * d * 4) as u64;
    let c_bytes = (c * d * 4) as u64;
    let old_bytes = (old_glu * d * 4) as u64;

    // ---- macaron FF1 over the new frames ----
    let input_new = batch.alloc(act);
    match input_gpu {
        Some(src) => batch.copy(src, 0, &input_new, 0, act),
        None => batch.write(
            &input_new,
            bytemuck_safe(&input[rel(s) * d..rel(s) * d + c * d]),
        ),
    }
    let ln = batch.alloc(act);
    kern.ln.record(
        batch,
        &input_new,
        0,
        &b.norm_ff1.w,
        &b.norm_ff1.b,
        c,
        d,
        b.norm_ff1.eps,
        &ln,
    );
    let h = batch.alloc(dff_size);
    kern.gemm.record(
        batch,
        &b.ff1_lin1.packed,
        &b.ff1_lin1.q,
        &b.ff1_lin1.s,
        b.ff1_lin1.bias.as_ref(),
        &ln,
        c,
        b.ff1_lin1.k,
        &h,
    );
    let hs = batch.alloc(dff_size);
    kern.ew.record_silu(batch, &h, &hs, c * dff);
    let f = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.ff1_lin2.packed,
        &b.ff1_lin2.q,
        &b.ff1_lin2.s,
        b.ff1_lin2.bias.as_ref(),
        &hs,
        c,
        b.ff1_lin2.k,
        &f,
    );
    let y = batch.alloc(act);
    kern.ew
        .record_add_mul(batch, &input_new, &f, &y, c * d, 0.5);

    // ---- rel-pos MHSA over the band ----
    let ln = batch.alloc(act);
    kern.ln.record(
        batch,
        &y,
        0,
        &b.norm_att.w,
        &b.norm_att.b,
        c,
        d,
        b.norm_att.eps,
        &ln,
    );
    let k_new = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.attn_k.packed,
        &b.attn_k.q,
        &b.attn_k.s,
        b.attn_k.bias.as_ref(),
        &ln,
        c,
        b.attn_k.k,
        &k_new,
    );
    let v_new = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.attn_v.packed,
        &b.attn_v.q,
        &b.attn_v.s,
        b.attn_v.bias.as_ref(),
        &ln,
        c,
        b.attn_v.k,
        &v_new,
    );
    let q_new = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.attn_q.packed,
        &b.attn_q.q,
        &b.attn_q.s,
        b.attn_q.bias.as_ref(),
        &ln,
        c,
        b.attn_q.k,
        &q_new,
    );
    // Combined band on the GPU: cached frames via host upload, new frames
    // copied from the scratch results above.
    let kv = batch.alloc(band_size);
    batch.write(&kv, bytemuck_safe(k_v_in));
    batch.copy(&k_new, 0, &kv, (k_v_in.len() * 4) as u64, c_bytes);
    let vv = batch.alloc(band_size);
    batch.write(&vv, bytemuck_safe(v_v_in));
    batch.copy(&v_new, 0, &vv, (v_v_in.len() * 4) as u64, c_bytes);
    let attn_out = batch.alloc(act);
    kern.attn.record(
        batch,
        &q_new,
        &kv,
        &vv,
        pos_p,
        &b.pos_u,
        &b.pos_v,
        c,
        d,
        m.n_heads,
        scale,
        s,
        k_lo,
        band,
        chunk,
        left_chunks,
        k_hi,
        chunk - 1,
        &attn_out,
    );
    let o = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.attn_out.packed,
        &b.attn_out.q,
        &b.attn_out.s,
        b.attn_out.bias.as_ref(),
        &attn_out,
        c,
        b.attn_out.k,
        &o,
    );
    let y2 = batch.alloc(act);
    kern.ew.record_add_mul(batch, &y, &o, &y2, c * d, 1.0);

    // ---- conv module ----
    // LN_conv + pw1 + GLU only for the c new frames; the old frames'
    // GLU is cached on the host and uploaded here only to feed the dw
    // window, so the per-batch cost does not scale with `conv_left`.
    let lnc = batch.alloc(act);
    kern.ln.record(
        batch,
        &y2,
        0,
        &b.norm_conv.w,
        &b.norm_conv.b,
        c,
        d,
        b.norm_conv.eps,
        &lnc,
    );
    let h3 = batch.alloc(pw1_size);
    kern.gemm.record(
        batch,
        &b.pw1.packed,
        &b.pw1.q,
        &b.pw1.s,
        b.pw1.bias.as_ref(),
        &lnc,
        c,
        b.pw1.k,
        &h3,
    );
    let glu_new = batch.alloc(act);
    kern.ew.record_glu(batch, &h3, &glu_new, c * d, d);
    // dw window: cached old GLU (real values) + new GLU; the kernel's
    // left pad only affects outputs below `old_glu`, which are
    // discarded.
    let glu = batch.alloc(glu_act);
    batch.write(&glu, bytemuck_safe(&glu_in[rel(conv_lo) * d..rel(s) * d]));
    batch.copy(&glu_new, 0, &glu, old_bytes, act);
    let conv = batch.alloc(glu_act);
    kern.dw
        .record(batch, &glu, &b.dw, n_glu, d, b.dw_kh, b.dw_pad_left, &conv);
    let ln2 = batch.alloc(act);
    kern.ln.record(
        batch,
        &conv,
        old_bytes,
        &b.conv_ln.w,
        &b.conv_ln.b,
        c,
        d,
        b.conv_ln.eps,
        &ln2,
    );
    let conv2 = batch.alloc(act);
    kern.ew.record_silu(batch, &ln2, &conv2, c * d);
    let o2 = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.pw2.packed,
        &b.pw2.q,
        &b.pw2.s,
        b.pw2.bias.as_ref(),
        &conv2,
        c,
        b.pw2.k,
        &o2,
    );
    let y3 = batch.alloc(act);
    kern.ew.record_add_mul(batch, &y2, &o2, &y3, c * d, 1.0);

    // ---- macaron FF2 ----
    let ln = batch.alloc(act);
    kern.ln.record(
        batch,
        &y3,
        0,
        &b.norm_ff2.w,
        &b.norm_ff2.b,
        c,
        d,
        b.norm_ff2.eps,
        &ln,
    );
    let h5 = batch.alloc(dff_size);
    kern.gemm.record(
        batch,
        &b.ff2_lin1.packed,
        &b.ff2_lin1.q,
        &b.ff2_lin1.s,
        b.ff2_lin1.bias.as_ref(),
        &ln,
        c,
        b.ff2_lin1.k,
        &h5,
    );
    let h5s = batch.alloc(dff_size);
    kern.ew.record_silu(batch, &h5, &h5s, c * dff);
    let f2 = batch.alloc(act);
    kern.gemm.record(
        batch,
        &b.ff2_lin2.packed,
        &b.ff2_lin2.q,
        &b.ff2_lin2.s,
        b.ff2_lin2.bias.as_ref(),
        &h5s,
        c,
        b.ff2_lin2.k,
        &f2,
    );
    let y4 = batch.alloc(act);
    kern.ew.record_add_mul(batch, &y3, &f2, &y4, c * d, 0.5);

    // ---- final per-block LN ----
    let out = batch.alloc(act);
    kern.ln.record(
        batch,
        &y4,
        0,
        &b.norm_out.w,
        &b.norm_out.b,
        c,
        d,
        b.norm_out.eps,
        &out,
    );

    // No submit here: the caller submits once after all blocks and pulls
    // back the persistent results with one batched download.
    (out, glu_new, k_new, v_new)
}

impl crate::streaming::StreamEncoder for GpuStreamingEncoder {
    fn encode_new(
        &mut self,
        enc: &mut Encoder,
        mel: &[f32],
        n_mels: usize,
        t0: usize,
        t1: usize,
        prompt_id: Option<u32>,
        fin: bool,
    ) -> anyhow::Result<()> {
        GpuStreamingEncoder::encode_new(
            self, enc, mel, n_mels, t0, t1, prompt_id, fin,
        )
    }

    fn frames(&self, from: usize, to: usize) -> anyhow::Result<&[f32]> {
        GpuStreamingEncoder::frames(self, from, to)
    }

    fn total(&self) -> usize {
        self.total
    }
}

/// Try to build a GPU streaming encoder for `enc`, uploading the weights
/// to the device. Returns `Ok(None)` when no GPU is available, so the
/// caller can transparently fall back to the CPU encoder; `Err` is
/// reserved for the case where a GPU exists but initialization fails.
pub fn try_build(
    enc: &Encoder,
) -> anyhow::Result<Option<Box<dyn crate::streaming::StreamEncoder>>> {
    let Some(ctx) = super::context::GpuContext::init() else {
        return Ok(None);
    };
    let ctx = Arc::new(ctx);
    let model = super::model::GpuModel::from_encoder(&ctx, enc)?;
    let senc = GpuStreamingEncoder::new(&ctx, model)?;
    Ok(Some(Box::new(senc)))
}
