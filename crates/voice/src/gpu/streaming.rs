//! GPU incremental streaming encoder.
//!
//! Mirrors `crate::nemotron::streaming::StreamingEncoder`: the same
//! per-frame caches (`pre`, per-block outputs, pre-FF2 activations, band
//! K/V) and the same chunk-aligned attention band, so the output for any
//! frame equals the offline GPU encode. The heavy per-block work (GEMMs,
//! LayerNorms, streaming attention, dw conv) runs on the GPU via the
//! standalone kernels; pre_encode and the tiny prompt MLP stay on the
//! CPU (convs + 2-layer MLP are small). Caches are host-side for now —
//! moving them to persistent GPU buffers is a later optimization.

use std::sync::Arc;

use anyhow::{Result, bail};

use super::context::GpuContext;
use super::kernels::{
    AttnStreamKernel, DwConvKernel, ElementwiseKernel, LayerNormKernel, Q8Gemm,
};
use super::model::{GpuBlock, GpuModel};
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
        let last = self
            .blocks
            .last()
            .ok_or_else(|| anyhow::anyhow!("gpu streaming encoder: no blocks"))?;
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
        let mut out_new: Vec<f32> = Vec::new();
        for b in 0..n_layers {
            let input: &[f32] = if b == 0 {
                &self.pre
            } else {
                &self.blocks[b - 1]
            };
            let (no, nco, k_new, v_new) = block_new(
                &mut self.kern,
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
/// frames `[kv_lo, s)`. Returns `(out, nco, k_new, v_new)`.
#[allow(clippy::too_many_arguments)]
fn block_new(
    kern: &mut BlockKernels,
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

    // ---- macaron FF1 over the new frames ----
    let input_new = &input[rel(s) * d..rel(s) * d + c * d];
    let ln = kern.ln.forward(
        input_new,
        &b.norm_ff1.w,
        &b.norm_ff1.b,
        c,
        d,
        b.norm_ff1.eps,
    );
    let mut h = kern.gemm.gemm(
        &b.ff1_lin1.packed,
        &b.ff1_lin1.q,
        &b.ff1_lin1.s,
        b.ff1_lin1.bias.as_ref(),
        &ln,
        c,
        b.ff1_lin1.k,
    );
    h = kern.ew.silu(&h);
    let f = kern.gemm.gemm(
        &b.ff1_lin2.packed,
        &b.ff1_lin2.q,
        &b.ff1_lin2.s,
        b.ff1_lin2.bias.as_ref(),
        &h,
        c,
        b.ff1_lin2.k,
    );
    let mut y = kern.ew.add_mul(input_new, &f, 0.5);

    // ---- rel-pos MHSA over the band ----
    let ln = kern.ln.forward(&y, &b.norm_att.w, &b.norm_att.b, c, d, b.norm_att.eps);
    let k_new = kern.gemm.gemm(
        &b.attn_k.packed,
        &b.attn_k.q,
        &b.attn_k.s,
        b.attn_k.bias.as_ref(),
        &ln,
        c,
        b.attn_k.k,
    );
    let v_new = kern.gemm.gemm(
        &b.attn_v.packed,
        &b.attn_v.q,
        &b.attn_v.s,
        b.attn_v.bias.as_ref(),
        &ln,
        c,
        b.attn_v.k,
    );
    let q_new = kern.gemm.gemm(
        &b.attn_q.packed,
        &b.attn_q.q,
        &b.attn_q.s,
        b.attn_q.bias.as_ref(),
        &ln,
        c,
        b.attn_q.k,
    );
    let band = k_hi - k_lo;
    let mut kv = Vec::with_capacity(band * d);
    kv.extend_from_slice(k_v_in);
    kv.extend_from_slice(&k_new);
    let mut vv = Vec::with_capacity(band * d);
    vv.extend_from_slice(v_v_in);
    vv.extend_from_slice(&v_new);
    let attn_out = kern.attn.forward(
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
    );
    let o = kern.gemm.gemm(
        &b.attn_out.packed,
        &b.attn_out.q,
        &b.attn_out.s,
        b.attn_out.bias.as_ref(),
        &attn_out,
        c,
        b.attn_out.k,
    );
    y = kern.ew.add_mul(&y, &o, 1.0);

    // ---- conv module ----
    let nco = y.clone();
    let mut lnc_in = vec![0.0f32; n_glu * d];
    lnc_in[..old_glu * d]
        .copy_from_slice(&pre_conv_in[rel(conv_lo) * d..rel(conv_lo) * d + old_glu * d]);
    lnc_in[old_glu * d..].copy_from_slice(&y);
    let lnc = kern.ln.forward(
        &lnc_in,
        &b.norm_conv.w,
        &b.norm_conv.b,
        n_glu,
        d,
        b.norm_conv.eps,
    );
    let h3 = kern.gemm.gemm(
        &b.pw1.packed,
        &b.pw1.q,
        &b.pw1.s,
        b.pw1.bias.as_ref(),
        &lnc,
        n_glu,
        b.pw1.k,
    );
    let glu = kern.ew.glu(&h3, d);
    let conv = kern
        .dw
        .forward(&glu, &b.dw, n_glu, d, b.dw_kh, b.dw_pad_left);
    let ln2 = kern.ln.forward(
        &conv[old_glu * d..],
        &b.conv_ln.w,
        &b.conv_ln.b,
        c,
        d,
        b.conv_ln.eps,
    );
    let conv2 = kern.ew.silu(&ln2);
    let o2 = kern.gemm.gemm(
        &b.pw2.packed,
        &b.pw2.q,
        &b.pw2.s,
        b.pw2.bias.as_ref(),
        &conv2,
        c,
        b.pw2.k,
    );
    y = kern.ew.add_mul(&y, &o2, 1.0);

    // ---- macaron FF2 ----
    let ln = kern.ln.forward(&y, &b.norm_ff2.w, &b.norm_ff2.b, c, d, b.norm_ff2.eps);
    let mut h5 = kern.gemm.gemm(
        &b.ff2_lin1.packed,
        &b.ff2_lin1.q,
        &b.ff2_lin1.s,
        b.ff2_lin1.bias.as_ref(),
        &ln,
        c,
        b.ff2_lin1.k,
    );
    h5 = kern.ew.silu(&h5);
    let f2 = kern.gemm.gemm(
        &b.ff2_lin2.packed,
        &b.ff2_lin2.q,
        &b.ff2_lin2.s,
        b.ff2_lin2.bias.as_ref(),
        &h5,
        c,
        b.ff2_lin2.k,
    );
    let y2 = kern.ew.add_mul(&y, &f2, 0.5);

    // ---- final per-block LN ----
    let out = kern.ln.forward(
        &y2,
        &b.norm_out.w,
        &b.norm_out.b,
        c,
        d,
        b.norm_out.eps,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::nemotron::streaming::StreamingEncoder;
    use crate::nemotron::Nemotron;

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
        let model = GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");
        let mut seed = 0x1234_abcd_5678_ef01u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let n_batches = 3;
        let mel: Vec<f32> = (0..n_batches * 64 * n_mels).map(|_| rnd_f()).collect();

        // Offline GPU encode of the whole audio, then take frames [0, 24).
        let mut genc = crate::gpu::encoder::GpuEncoder::new(&ctx).expect("gpu encoder");
        let x = cpu.encoder.pre_encode_forward(&mel, mel.len() / n_mels, n_mels);
        let t_enc = x.len() / d;
        let pe = cpu.encoder.pos_emb(t_enc);
        let off = genc.encode_blocks(&model, &x, t_enc, &pe);
        assert!(t_enc >= 24, "offline t_enc={t_enc} too small");
        let off = &off[..24 * d];

        let mut senc = GpuStreamingEncoder::new(&ctx, model).expect("build kernels");
        let mut stream_out = Vec::new();
        for bi in 0..n_batches {
            let t0 = bi * 64;
            senc
                .encode_new(&mut cpu.encoder, &mel, n_mels, t0, t0 + 64, None)
                .expect("gpu stream");
            let s = t0 / 8;
            let c = 8;
            stream_out.extend_from_slice(senc.frames(s, s + c).expect("frames"));
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
        let model = GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");

        // Deterministic mel: 3 batches of BATCH_MEL=64 mel frames.
        let mut seed = 0x1234_abcd_5678_ef01u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let n_batches = 3;
        let mel: Vec<f32> = (0..n_batches * 64 * n_mels).map(|_| rnd_f()).collect();

        // Phase 1: offline reference divergence per batch (CPU vs GPU,
        // the inherent f32-vs-i8 noise of this audio), before the model
        // moves into the streaming encoder.
        let mut genc = crate::gpu::encoder::GpuEncoder::new(&ctx).expect("gpu encoder");
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
            let x = cpu.encoder.pre_encode_forward(&mel[..t1 * n_mels], t1, n_mels);
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
        let mut senc_gpu = GpuStreamingEncoder::new(&ctx, model).expect("build kernels");
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
