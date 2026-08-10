//! GPU encoder: one Conformer block per call, orchestrated over the
//! standalone kernels. Activations round-trip through host memory between
//! ops (correctness first; the persistent-buffer batched pipeline is a
//! later optimization).

use std::sync::Arc;

use anyhow::Result;

use super::context::GpuContext;
use super::kernels::{
    AttentionKernel, DwConvKernel, ElementwiseKernel, LayerNormKernel, Q8Gemm,
};
use super::model::{GpuBlock, GpuModel, LinBufs, NormW};

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

    fn gemm_lin(&mut self, lin: &LinBufs, x: &[f32], t: usize) -> Vec<f32> {
        self.gemm.gemm(&lin.packed, &lin.q, &lin.s, lin.bias.as_ref(), x, t, lin.k)
    }

    fn apply_ln(&mut self, n: &NormW, x: &[f32], t: usize) -> Vec<f32> {
        self.ln.forward(x, &n.w, &n.b, t, n.d, n.eps)
    }

    /// One Conformer block, mirroring `Encoder::block_forward` exactly.
    /// `x` is `[t, d]`; `pe` is the positional embedding `[2t-1, d]`.
    /// Returns the block output `[t, d]` (after the per-block LN).
    pub fn block_forward(
        &mut self,
        m: &GpuModel,
        b: &GpuBlock,
        x: &[f32],
        t: usize,
        pe: &[f32],
    ) -> Vec<f32> {
        let d = m.d;
        let scale = 1.0 / (m.head_dim as f32).sqrt();

        // ---- macaron FF1 ----
        let ln = self.apply_ln(&b.norm_ff1, x, t);
        let mut h = self.gemm_lin(&b.ff1_lin1, &ln, t);
        h = self.ew.silu(&h);
        let f = self.gemm_lin(&b.ff1_lin2, &h, t);
        let mut y = self.ew.add_mul(x, &f, 0.5);

        // ---- rel-pos MHSA ----
        let ln = self.apply_ln(&b.norm_att, &y, t);
        let q = self.gemm_lin(&b.attn_q, &ln, t);
        let k = self.gemm_lin(&b.attn_k, &ln, t);
        let v = self.gemm_lin(&b.attn_v, &ln, t);
        let p = self.gemm_lin(&b.attn_pos, pe, 2 * t - 1);
        let attn_out = self.attn.forward(
            &q,
            &k,
            &v,
            &b.pos_u,
            &b.pos_v,
            &p,
            t,
            d,
            m.n_heads,
            scale,
            m.left,
            m.right,
        );
        let o = self.gemm_lin(&b.attn_out, &attn_out, t);
        y = self.ew.add_mul(&y, &o, 1.0);

        // ---- conv module ----
        let ln = self.apply_ln(&b.norm_conv, &y, t);
        let h3 = self.gemm_lin(&b.pw1, &ln, t);
        let glu = self.ew.glu(&h3, d);
        let conv = self.dw.forward(&glu, &b.dw, t, d, b.dw_kh, b.dw_pad_left);
        let ln = self.apply_ln(&b.conv_ln, &conv, t);
        let conv = self.ew.silu(&ln);
        let o2 = self.gemm_lin(&b.pw2, &conv, t);
        y = self.ew.add_mul(&y, &o2, 1.0);

        // ---- macaron FF2 ----
        let ln = self.apply_ln(&b.norm_ff2, &y, t);
        let mut h5 = self.gemm_lin(&b.ff2_lin1, &ln, t);
        h5 = self.ew.silu(&h5);
        let f2 = self.gemm_lin(&b.ff2_lin2, &h5, t);
        let y2 = self.ew.add_mul(&y, &f2, 0.5);

        // ---- final per-block LN ----
        self.apply_ln(&b.norm_out, &y2, t)
    }

    /// Run all conformer blocks over the pre-encoded `x` (`[t, d]`).
    pub fn encode_blocks(
        &mut self,
        m: &GpuModel,
        x: &[f32],
        t: usize,
        pe: &[f32],
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        for b in &m.blocks {
            out = self.block_forward(m, b, &out, t, pe);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::nemotron::Nemotron;

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
        let model = GpuModel::from_encoder(&ctx, &cpu.encoder).expect("upload weights");
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
            let norm: f32 = ref_row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
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
