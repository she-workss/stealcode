//! GPU incremental streaming encoder.
//!
//! Mirrors `crate::nemotron::streaming::StreamingEncoder`: the same
//! per-frame caches (`pre`, per-block outputs, pre-FF2 activations, band
//! K/V) and the same chunk-aligned attention band, so the output for any
//! frame equals the offline GPU encode. The heavy per-block work (GEMMs,
//! LayerNorms, streaming attention, dw conv) runs on the GPU: all of a
//! block's dispatches are recorded into one [`ComputeBatch`] and submitted
//! together, keeping intermediate activations on the GPU. pre_encode and
//! the tiny prompt MLP stay on the CPU (convs + 2-layer MLP are small).
//! Caches are host-side for now — moving them to persistent GPU buffers is
//! a later optimization.

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
use crate::nemotron::encoder::Encoder;

/// Kernel handles shared by every block.
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
pub struct GpuStreamingEncoder {
    kern: BlockKernels,
    model: GpuModel,
    /// pre_encode output cache.
    pre: Vec<f32>,
    /// Per-block output caches; `blocks[b]` feeds block `b+1`.
    blocks: Vec<Vec<f32>>,
    /// Per-block pre-FF2 activations (needed for the conv module's
    /// causal left context).
    pre_conv: Vec<Vec<f32>>,
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
            pre_conv: vec![Vec::new(); n_blocks],
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
        let mut batch = ComputeBatch::new(&self.model.ctx);
        let mut out_new: Vec<f32> = Vec::new();
        for b in 0..n_layers {
            let input: &[f32] = if b == 0 {
                &self.pre
            } else {
                &self.blocks[b - 1]
            };
            let (no, nco, k_new, v_new) = block_new(
                &mut self.kern,
                &mut batch,
                &self.model,
                &self.model.blocks[b],
                d,
                input,
                &self.pre_conv[b],
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
            self.blocks[b].extend_from_slice(&no);
            self.pre_conv[b].extend_from_slice(&nco);
            self.k_v[b].extend_from_slice(&k_new);
            self.v_v[b].extend_from_slice(&v_new);
            out_new = no;
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
            for v in &mut self.pre_conv {
                v.drain(..drop);
            }
            self.base = keep;
        }
        Ok(())
    }
}

/// Compute one conformer block's outputs for frames `[s, t_new)`
/// (absolute indices) on the GPU. `input` is the block's input cache
/// (frames `[base, t_new)`, old + new), `pre_conv_in` the block's cached
/// pre-FF2 activations, `k_v_in`/`v_v_in` the block's cached K/V for band
/// frames `[kv_lo, s)`. All dispatches are recorded into `batch` and
/// submitted once; the persistent results (`out`, `nco`, `k_new`,
/// `v_new`) are downloaded after the submit. Returns `(out, nco,
/// k_new, v_new)`.
#[allow(clippy::too_many_arguments)]
fn block_new(
    kern: &mut BlockKernels,
    batch: &mut ComputeBatch,
    m: &GpuModel,
    b: &GpuBlock,
    d: usize,
    input: &[f32],
    pre_conv_in: &[f32],
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
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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
    let pw1_size = (n_glu * b.pw1.packed.rows * 4) as u64;
    let band = k_hi - k_lo;
    let band_size = (band * d * 4) as u64;
    let c_bytes = (c * d * 4) as u64;
    let old_bytes = (old_glu * d * 4) as u64;

    // ---- macaron FF1 over the new frames ----
    let input_new = batch.alloc(act);
    batch.write(
        &input_new,
        bytemuck_safe(&input[rel(s) * d..rel(s) * d + c * d]),
    );
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
        3,
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

    // Save the pre-FF2 activation (nco) before it is overwritten.
    let nco_b = batch.alloc(act);
    batch.copy(&y2, 0, &nco_b, 0, act);

    // ---- conv module ----
    let lnc_in = batch.alloc(glu_act);
    batch.write(
        &lnc_in,
        bytemuck_safe(
            &pre_conv_in[rel(conv_lo) * d..rel(conv_lo) * d + old_glu * d],
        ),
    );
    batch.copy(&y2, 0, &lnc_in, old_bytes, act);
    let lnc = batch.alloc(glu_act);
    kern.ln.record(
        batch,
        &lnc_in,
        0,
        &b.norm_conv.w,
        &b.norm_conv.b,
        n_glu,
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
        n_glu,
        b.pw1.k,
        &h3,
    );
    let glu = batch.alloc(glu_act);
    kern.ew.record_glu(batch, &h3, &glu, n_glu * d, d);
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

    // Submit once, then pull back the small persistent results.
    batch.submit();
    let out = bytes_to_f32(&m.ctx.download(&out, act), c * d);
    let nco = bytes_to_f32(&m.ctx.download(&nco_b, act), c * d);
    let k_new = bytes_to_f32(&m.ctx.download(&k_new, act), c * d);
    let v_new = bytes_to_f32(&m.ctx.download(&v_new, act), c * d);
    (out, nco, k_new, v_new)
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

impl crate::nemotron::streaming::StreamEncoder for GpuStreamingEncoder {
    fn encode_new(
        &mut self,
        enc: &mut Encoder,
        mel: &[f32],
        n_mels: usize,
        t0: usize,
        t1: usize,
        prompt_id: Option<u32>,
    ) -> anyhow::Result<()> {
        GpuStreamingEncoder::encode_new(self, enc, mel, n_mels, t0, t1, prompt_id)
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
) -> anyhow::Result<Option<Box<dyn crate::nemotron::streaming::StreamEncoder>>> {
    let Some(ctx) = super::context::GpuContext::init() else {
        return Ok(None);
    };
    let ctx = Arc::new(ctx);
    let model = super::model::GpuModel::from_encoder(&ctx, enc)?;
    let senc = GpuStreamingEncoder::new(&ctx, model)?;
    Ok(Some(Box::new(senc)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gpu::GpuContext,
        nemotron::{Nemotron, streaming::StreamingEncoder},
    };

    #[test]
    fn gpu_streaming_matches_gpu_offline() {
        let path = crate::default_model_path();
        if !path.exists() {
            eprintln!("skipping: model not found at {}", path.display());
            return;
        }
        let mut cpu = Nemotron::load(&path).expect("load model");
        let d = cpu.encoder.cfg.d_model;
        let n_mels = cpu.encoder.cfg.feat_in;
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let model =
            GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");
        let mut seed = 0x1234_abcd_5678_ef01u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let n_batches = 3;
        let mel: Vec<f32> =
            (0..n_batches * 64 * n_mels).map(|_| rnd_f()).collect();

        // Offline GPU encode of the whole audio, then take frames [0, 24).
        let mut genc =
            crate::gpu::encoder::GpuEncoder::new(&ctx).expect("gpu encoder");
        let x =
            cpu.encoder
                .pre_encode_forward(&mel, mel.len() / n_mels, n_mels);
        let t_enc = x.len() / d;
        let pe = cpu.encoder.pos_emb(t_enc);
        let off = genc.encode_blocks(&model, &x, t_enc, &pe);
        assert!(t_enc >= 24, "offline t_enc={t_enc} too small");
        let off = &off[..24 * d];

        let mut senc =
            GpuStreamingEncoder::new(&ctx, model).expect("build kernels");
        let mut stream_out = Vec::new();
        for bi in 0..n_batches {
            let t0 = bi * 64;
            senc.encode_new(&mut cpu.encoder, &mel, n_mels, t0, t0 + 64, None)
                .expect("gpu stream");
            let s = t0 / 8;
            let c = 8;
            stream_out
                .extend_from_slice(senc.frames(s, s + c).expect("frames"));
        }

        let mut worst = 0.0f32;
        for i in 0..stream_out.len() {
            worst = worst.max((stream_out[i] - off[i]).abs());
        }
        assert!(
            worst < 1e-5,
            "GPU streaming != GPU offline: max abs diff={worst:.3e}"
        );
    }

    #[test]
    fn mem_diag() {
        let path = crate::default_model_path();
        if !path.exists() {
            eprintln!("skipping: model not found at {}", path.display());
            return;
        }
        #[cfg(target_os = "windows")]
        fn ws_mb() -> f64 {
            use std::ffi::c_void;
            #[repr(C)]
            struct PMC {
                cb: u32,
                _pf: u32,
                peak_ws: usize,
                ws: usize,
                _x: [usize; 6],
                pagefile: usize,
                _peak_pf: usize,
            }
            unsafe extern "system" {
                fn GetCurrentProcess() -> *mut c_void;
                fn K32GetProcessMemoryInfo(
                    p: *mut c_void,
                    info: *mut PMC,
                    cb: u32,
                ) -> i32;
            }
            unsafe {
                let mut pmc: PMC = std::mem::zeroed();
                pmc.cb = std::mem::size_of::<PMC>() as u32;
                let p = GetCurrentProcess();
                K32GetProcessMemoryInfo(p, &mut pmc, pmc.cb);
                let ws = pmc.ws as f64 / (1024.0 * 1024.0);
                let commit = pmc.pagefile as f64 / (1024.0 * 1024.0);
                eprintln!("  [mem] WS={ws:.1} MB commit={commit:.1} MB");
                ws
            }
        }
        #[cfg(not(target_os = "windows"))]
        fn ws_mb() -> f64 {
            0.0
        }
        let m = ws_mb();
        eprintln!("[mem] before load: {m:.1} MB");
        let Some(ctx) = super::super::context::GpuContext::init() else {
            eprintln!("[mem] no GPU; CPU fallback");
            return;
        };
        let ctx = Arc::new(ctx);
        eprintln!("[mem] after gpu init (no model): {:.1} MB", ws_mb());
        let mut cpu = crate::nemotron::Nemotron::load(&path).expect("load");
        eprintln!("[mem] after cpu load: {:.1} MB", ws_mb());
        let model =
            GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        eprintln!("[mem] after upload + poll: {:.1} MB", ws_mb());
        cpu.encoder.blocks = Vec::new();
        eprintln!("[mem] after free cpu blocks: {:.1} MB", ws_mb());
        drop(model);
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        eprintln!("[mem] after drop(gpu model): {:.1} MB", ws_mb());
        drop(cpu);
        eprintln!("[mem] after drop(whole cpu): {:.1} MB", ws_mb());
        drop(ctx);
        eprintln!("[mem] after drop(ctx): {:.1} MB", ws_mb());
    }

    #[test]
    fn gpu_streaming_matches_cpu_streaming() {
        let path = crate::default_model_path();
        if !path.exists() {
            eprintln!("skipping: model not found at {}", path.display());
            return;
        }
        let mut cpu = Nemotron::load(&path).expect("load model");
        let d = cpu.encoder.cfg.d_model;
        let n_mels = cpu.encoder.cfg.feat_in;
        let n_layers = cpu.encoder.cfg.n_layers;

        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let model =
            GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");

        // Deterministic mel: 3 batches of BATCH_MEL=64 mel frames.
        let mut seed = 0x1234_abcd_5678_ef01u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let n_batches = 3;
        let mel: Vec<f32> =
            (0..n_batches * 64 * n_mels).map(|_| rnd_f()).collect();

        // Phase 1: offline reference divergence per batch (CPU vs GPU,
        // the inherent f32-vs-i8 noise of this audio), before the model
        // moves into the streaming encoder.
        let mut genc =
            crate::gpu::encoder::GpuEncoder::new(&ctx).expect("gpu encoder");
        let mut offline_err = vec![0.0f32; n_batches];
        for bi in 0..n_batches {
            let t1 = (bi + 1) * 64;
            let s = bi * 8;
            let c = 8;
            let mut cpu_out = Vec::new();
            let _ = cpu
                .encoder
                .encode(&mel[..t1 * n_mels], t1, None, &mut cpu_out)
                .expect("cpu offline");
            let x =
                cpu.encoder
                    .pre_encode_forward(&mel[..t1 * n_mels], t1, n_mels);
            let t_enc = x.len() / d;
            let pe = cpu.encoder.pos_emb(t_enc);
            let gpu_out = genc.encode_blocks(&model, &x, t_enc, &pe);
            for tt in 0..c {
                let o_norm: f32 = cpu_out[(s + tt) * d..(s + tt + 1) * d]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let o_err: f32 = cpu_out[(s + tt) * d..(s + tt + 1) * d]
                    .iter()
                    .zip(&gpu_out[(s + tt) * d..(s + tt + 1) * d])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                offline_err[bi] = offline_err[bi].max(o_err / o_norm);
            }
        }

        // Phase 2: streaming, then require the GPU-vs-CPU streaming
        // divergence to stay within the offline path's own noise.
        let mut senc_cpu = StreamingEncoder::new(n_layers);
        let mut senc_gpu =
            GpuStreamingEncoder::new(&ctx, model).expect("build kernels");
        for bi in 0..n_batches {
            let t0 = bi * 64;
            let t1 = t0 + 64;
            senc_cpu
                .encode_new(&mut cpu.encoder, &mel, n_mels, t0, t1, None)
                .expect("cpu stream");
            senc_gpu
                .encode_new(&mut cpu.encoder, &mel, n_mels, t0, t1, None)
                .expect("gpu stream");
            let s = t0 / 8;
            let c = 64 / 8;
            let cf = senc_cpu.frames(s, s + c).expect("cpu frames");
            let gf = senc_gpu.frames(s, s + c).expect("gpu frames");

            let mut stream_err = 0.0f32;
            for tt in 0..c {
                let norm: f32 = cf[tt * d..(tt + 1) * d]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let err: f32 = cf[tt * d..(tt + 1) * d]
                    .iter()
                    .zip(&gf[tt * d..(tt + 1) * d])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                stream_err = stream_err.max(err / norm);
            }
            eprintln!(
                "batch {bi}: stream_rel_err={stream_err:.3e} offline_rel_err={:.3e}",
                offline_err[bi]
            );
            assert!(
                stream_err <= offline_err[bi] * 2.0 + 1e-3,
                "batch {bi}: streaming divergence ({stream_err:.3e}) exceeds \
                 offline f32-i8 divergence ({:.3e})",
                offline_err[bi]
            );
        }
    }
}
