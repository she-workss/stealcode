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
        batch: &mut ComputeBatch,
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
        batch: &mut ComputeBatch,
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
        batch: &mut ComputeBatch,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gpu::GpuContext, nemotron::Nemotron};

    #[test]
    fn gpu_blocks_match_cpu_encoder() {
        let path = crate::default_model_path();
        if !path.exists() {
            eprintln!("skipping: model not found at {}", path.display());
            return;
        }
        let mut cpu = Nemotron::load(&path).expect("load model");
        let d = cpu.encoder.cfg.d_model;
        let n_mels = cpu.encoder.cfg.feat_in;
        let t_mel = 40;

        // Real-ish mel input (deterministic noise is fine for parity).
        let mut seed = 0xabcd_ef01_2345_6789u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let mel: Vec<f32> = (0..t_mel * n_mels).map(|_| rnd_f()).collect();

        // CPU reference: full offline encode (pre_encode + all blocks).
        let mut cpu_out = Vec::new();
        let t_enc = cpu
            .encoder
            .encode(&mel, t_mel, None, &mut cpu_out)
            .expect("cpu encode");
        assert_eq!(cpu_out.len(), t_enc * d);
        assert!(t_enc <= 16, "unexpected t_enc={t_enc}");

        // GPU: same pre-encoded input, then run all blocks on GPU.
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let model =
            GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");
        let x = cpu.encoder.pre_encode_forward(&mel, t_mel, n_mels);
        let pe = cpu.encoder.pos_emb(t_enc);
        let mut enc = GpuEncoder::new(&ctx).expect("build kernels");
        let gpu_out = enc.encode_blocks(&model, &x, t_enc, &pe);

        // i8x i8 CPU vs f32x i8 GPU differ by activation-quantization
        // noise; tolerance is on a per-frame relative basis.
        let mut worst = 0.0f32;
        let mut worst_norm = 0.0f32;
        for tt in 0..t_enc {
            let ref_row = &cpu_out[tt * d..(tt + 1) * d];
            let gpu_row = &gpu_out[tt * d..(tt + 1) * d];
            let norm: f32 =
                ref_row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let err: f32 = ref_row
                .iter()
                .zip(gpu_row)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            worst = worst.max(err / norm);
            worst_norm = worst_norm.max(norm);
        }
        eprintln!(
            "t_enc={t_enc} blocks={} worst_rel_err={worst:.3e} row_norm~{worst_norm:.1}",
            model.blocks.len()
        );
        assert!(
            worst < 2e-2,
            "GPU encoder too far from CPU: worst_rel_err={worst:.3e}"
        );
    }
}
