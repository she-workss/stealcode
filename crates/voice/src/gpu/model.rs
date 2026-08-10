//! GPU-resident model weights, packed and uploaded from the CPU `Encoder`.
//!
//! Weight matrices are packed into the `q8_gemm` GPU layout once and kept
//! in storage buffers; per-block norm/biases stay on the host for now
//! (they are tiny and round-trip cheaply). The GEMM weight buffers are the
//! big ones and never leave the GPU after `from_encoder`.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::context::GpuContext;
use super::kernels::{f32_bytes, pack_q8, PackedQ8};
use crate::nemotron::encoder::Encoder;
use crate::nemotron::weights::{LayerNorm, Lin};

/// Uploaded Q8 weight matrix + bias for one linear layer.
pub struct LinBufs {
    /// Packed layout descriptor (used for dispatch sizing).
    pub packed: PackedQ8,
    /// `packed.q` on the GPU (storage).
    pub q: wgpu::Buffer,
    /// `packed.s` on the GPU (storage).
    pub s: wgpu::Buffer,
    /// Optional bias `[out]` on the GPU (storage).
    pub bias: Option<wgpu::Buffer>,
    /// Input dim `k` (packed.rows = output dim `n`).
    pub k: usize,
}

/// Host copies of a LayerNorm's affine params (uploaded per op).
pub struct NormW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub d: usize,
    pub eps: f32,
}

pub struct GpuBlock {
    pub norm_ff1: NormW,
    pub ff1_lin1: LinBufs,
    pub ff1_lin2: LinBufs,
    pub norm_att: NormW,
    pub attn_q: LinBufs,
    pub attn_k: LinBufs,
    pub attn_v: LinBufs,
    pub attn_pos: LinBufs,
    pub attn_out: LinBufs,
    pub pos_u: Vec<f32>,
    pub pos_v: Vec<f32>,
    pub norm_conv: NormW,
    pub pw1: LinBufs,
    /// Depthwise conv weights `[d, kh]`.
    pub dw: Vec<f32>,
    pub dw_kh: usize,
    pub dw_pad_left: usize,
    pub conv_ln: NormW,
    pub pw2: LinBufs,
    pub norm_ff2: NormW,
    pub ff2_lin1: LinBufs,
    pub ff2_lin2: LinBufs,
    pub norm_out: NormW,
}

pub struct GpuModel {
    pub ctx: Arc<GpuContext>,
    pub d: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub left: usize,
    pub right: usize,
    pub blocks: Vec<GpuBlock>,
}

fn pack_lin(ctx: &Arc<GpuContext>, lin: &Lin) -> Result<LinBufs> {
    let qm = lin
        .q
        .as_ref()
        .ok_or_else(|| anyhow!("GPU model requires Q8 linear weights"))?;
    let row_len = qm.row_len();
    let block_bytes = qm.padded_row() / row_len.div_ceil(32);
    let packed = pack_q8(qm.bytes(), qm.rows(), row_len, qm.padded_row(), block_bytes);
    let q = ctx.upload("voice/gpu lin q", &packed.q, wgpu::BufferUsages::STORAGE);
    let s = ctx.upload("voice/gpu lin s", &packed.s, wgpu::BufferUsages::STORAGE);
    let bias = lin
        .bias
        .as_ref()
        .map(|b| ctx.upload("voice/gpu lin bias", &f32_bytes(b), wgpu::BufferUsages::STORAGE));
    Ok(LinBufs {
        packed,
        q,
        s,
        bias,
        k: row_len,
    })
}

fn pack_norm(n: &LayerNorm) -> NormW {
    NormW {
        w: n.weight.clone(),
        b: n.bias.clone(),
        d: n.dim,
        eps: n.eps,
    }
}

impl GpuModel {
    /// Pack and upload every block's weights from an already-loaded CPU
    /// `Encoder`.
    pub fn from_encoder(ctx: &Arc<GpuContext>, enc: &Encoder) -> Result<Self> {
        let cfg = &enc.cfg;
        let head_dim = cfg.d_model / cfg.n_heads;
        let mut blocks = Vec::with_capacity(enc.blocks.len());
        for b in &enc.blocks {
            let dw = &b.dw;
            if dw.pad_right != 0 {
                return Err(anyhow!(
                    "GPU dwconv requires pad_right=0, got {}",
                    dw.pad_right
                ));
            }
            blocks.push(GpuBlock {
                norm_ff1: pack_norm(&b.norm_ff1),
                ff1_lin1: pack_lin(ctx, &b.ff1_lin1)?,
                ff1_lin2: pack_lin(ctx, &b.ff1_lin2)?,
                norm_att: pack_norm(&b.norm_att),
                attn_q: pack_lin(ctx, &b.attn_q)?,
                attn_k: pack_lin(ctx, &b.attn_k)?,
                attn_v: pack_lin(ctx, &b.attn_v)?,
                attn_pos: pack_lin(ctx, &b.attn_pos)?,
                attn_out: pack_lin(ctx, &b.attn_out)?,
                pos_u: b.pos_u.clone(),
                pos_v: b.pos_v.clone(),
                norm_conv: pack_norm(&b.norm_conv),
                pw1: pack_lin(ctx, &b.pw1)?,
                dw: dw.w.clone(),
                dw_kh: dw.kh,
                dw_pad_left: dw.pad_left,
                conv_ln: pack_norm(&b.conv_ln),
                pw2: pack_lin(ctx, &b.pw2)?,
                norm_ff2: pack_norm(&b.norm_ff2),
                ff2_lin1: pack_lin(ctx, &b.ff2_lin1)?,
                ff2_lin2: pack_lin(ctx, &b.ff2_lin2)?,
                norm_out: pack_norm(&b.norm_out),
            });
        }
        Ok(Self {
            ctx: ctx.clone(),
            d: cfg.d_model,
            n_heads: cfg.n_heads,
            head_dim,
            left: cfg.att_context_left,
            right: cfg.att_context_right,
            blocks,
        })
    }
}
