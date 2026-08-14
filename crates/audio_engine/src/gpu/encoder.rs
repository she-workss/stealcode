//! GPU encoder: one Conformer block per call, orchestrated over the
//! standalone kernels. All dispatches of a block are recorded into a
//! single [`ComputeBatch`] and submitted once, so activations stay on the
//! GPU; only the block output round-trips through host memory.

use std::sync::Arc;

use anyhow::Result;

use super::{
    batch::ComputeBatch,
    context::GpuContext,
    kernels::{
        AttentionKernel, DwConvKernel, ElementwiseKernel, LayerNormKernel,
        Q8Gemm, bytemuck_safe, bytes_to_f32,
    },
    model::{GpuBlock, GpuModel, LinBufs, NormW},
};

/// Kernel handles + per-block forward logic.
#[derive(Debug)]
pub struct GpuEncoder {
    gemm: Q8Gemm,
    ln: LayerNormKernel,
    ew: ElementwiseKernel,
    attn: AttentionKernel,
    dw: DwConvKernel,
}

impl GpuEncoder {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        Ok(Self {
            gemm: Q8Gemm::new(ctx)?,
            ln: LayerNormKernel::new(ctx)?,
            ew: ElementwiseKernel::new(ctx)?,
            attn: AttentionKernel::new(ctx)?,
            dw: DwConvKernel::new(ctx)?,
        })
    }

    fn gemm_lin(
        &mut self,
        batch: &mut ComputeBatch<'_>,
        lin: &LinBufs,
        x: &wgpu::Buffer,
        t: usize,
        out: &wgpu::Buffer,
    ) {
        self.gemm.record(
            batch,
            &lin.packed,
            &lin.q,
            &lin.s,
            lin.bias.as_ref(),
            x,
            t,
            lin.k,
            out,
        );
    }

    fn apply_ln(
        &mut self,
        batch: &mut ComputeBatch<'_>,
        n: &NormW,
        x: &wgpu::Buffer,
        x_off: u64,
        t: usize,
        out: &wgpu::Buffer,
    ) {
        self.ln
            .record(batch, x, x_off, &n.w, &n.b, t, n.d, n.eps, out);
    }

    /// One Conformer block, mirroring `Encoder::block_forward` exactly.
    /// `x` is `[t, d]`; `pe` is the positional embedding `[2t-1, d]`.
    /// Records every op into `batch`, submits once, and returns the block
    /// output `[t, d]` (after the per-block LN).
    pub fn block_forward(
        &mut self,
        batch: &mut ComputeBatch<'_>,
        m: &GpuModel,
        b: &GpuBlock,
        x: &[f32],
        t: usize,
        pe: &[f32],
    ) -> Vec<f32> {
        let d = m.d;
        let scale = 1.0 / (m.head_dim as f32).sqrt();
        let act = (t * d * 4) as u64;
        let dff = b.ff1_lin1.packed.rows;
        let dff_size = (t * dff * 4) as u64;
        let pw1_size = (t * b.pw1.packed.rows * 4) as u64;
        let pe_size = ((2 * t - 1) * d * 4) as u64;

        let xb = batch.alloc(act);
        batch.write(&xb, bytemuck_safe(x));
        let peb = batch.alloc(pe_size);
        batch.write(&peb, bytemuck_safe(pe));

        // ---- macaron FF1 ----
        let ln = batch.alloc(act);
        self.apply_ln(batch, &b.norm_ff1, &xb, 0, t, &ln);
        let h = batch.alloc(dff_size);
        self.gemm_lin(batch, &b.ff1_lin1, &ln, t, &h);
        let hs = batch.alloc(dff_size);
        self.ew.record_silu(batch, &h, &hs, t * dff);
        let f = batch.alloc(act);
        self.gemm_lin(batch, &b.ff1_lin2, &hs, t, &f);
        let y = batch.alloc(act);
        self.ew.record_add_mul(batch, &xb, &f, &y, t * d, 0.5);

        // ---- rel-pos MHSA ----
        let ln = batch.alloc(act);
        self.apply_ln(batch, &b.norm_att, &y, 0, t, &ln);
        let q = batch.alloc(act);
        self.gemm_lin(batch, &b.attn_q, &ln, t, &q);
        let k = batch.alloc(act);
        self.gemm_lin(batch, &b.attn_k, &ln, t, &k);
        let v = batch.alloc(act);
        self.gemm_lin(batch, &b.attn_v, &ln, t, &v);
        let p = batch.alloc(pe_size);
        self.gemm_lin(batch, &b.attn_pos, &peb, 2 * t - 1, &p);
        let attn_out = batch.alloc(act);
        self.attn.record(
            batch, &q, &k, &v, &b.pos_u, &b.pos_v, &p, t, d, m.n_heads, scale,
            m.left, m.right, &attn_out,
        );
        let o = batch.alloc(act);
        self.gemm_lin(batch, &b.attn_out, &attn_out, t, &o);
        let y2 = batch.alloc(act);
        self.ew.record_add_mul(batch, &y, &o, &y2, t * d, 1.0);

        // ---- conv module ----
        let ln = batch.alloc(act);
        self.apply_ln(batch, &b.norm_conv, &y2, 0, t, &ln);
        let h3 = batch.alloc(pw1_size);
        self.gemm_lin(batch, &b.pw1, &ln, t, &h3);
        let glu = batch.alloc(act);
        self.ew.record_glu(batch, &h3, &glu, t * d, d);
        let conv = batch.alloc(act);
        self.dw
            .record(batch, &glu, &b.dw, t, d, b.dw_kh, b.dw_pad_left, &conv);
        let ln = batch.alloc(act);
        self.apply_ln(batch, &b.conv_ln, &conv, 0, t, &ln);
        let conv = batch.alloc(act);
        self.ew.record_silu(batch, &ln, &conv, t * d);
        let o2 = batch.alloc(act);
        self.gemm_lin(batch, &b.pw2, &conv, t, &o2);
        let y3 = batch.alloc(act);
        self.ew.record_add_mul(batch, &y2, &o2, &y3, t * d, 1.0);

        // ---- macaron FF2 ----
        let ln = batch.alloc(act);
        self.apply_ln(batch, &b.norm_ff2, &y3, 0, t, &ln);
        let h5 = batch.alloc(dff_size);
        self.gemm_lin(batch, &b.ff2_lin1, &ln, t, &h5);
        let h5s = batch.alloc(dff_size);
        self.ew.record_silu(batch, &h5, &h5s, t * dff);
        let f2 = batch.alloc(act);
        self.gemm_lin(batch, &b.ff2_lin2, &h5s, t, &f2);
        let y4 = batch.alloc(act);
        self.ew.record_add_mul(batch, &y3, &f2, &y4, t * d, 0.5);

        // ---- final per-block LN ----
        let out = batch.alloc(act);
        self.apply_ln(batch, &b.norm_out, &y4, 0, t, &out);

        batch.submit();
        let bytes = m.ctx.download(&out, act);
        bytes_to_f32(&bytes, t * d)
    }

    /// Run all conformer blocks over the pre-encoded `x` (`[t, d]`).
    pub fn encode_blocks(
        &mut self,
        m: &GpuModel,
        x: &[f32],
        t: usize,
        pe: &[f32],
    ) -> Vec<f32> {
        let ctx = m.ctx.clone();
        let mut batch = ComputeBatch::new(&ctx);
        let mut out = x.to_vec();
        for b in &m.blocks {
            out = self.block_forward(&mut batch, m, b, &out, t, pe);
        }
        out
    }
}
