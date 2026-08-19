//! FastConformer encoder (cache-aware variant), offline one-shot path.
//!
//! Mirrors the C++ reference (`build_encoder_graph` +
//! `build_conformer_block` + `build_pre_encode`):
//!   * pre_encode: causal stride-2 subsampling stack (pad 2/1 on both spatial
//!     axes, conv with p=0), 128 -> 65 -> 33 -> 17 freq bins; flatten [t,
//!     17*256] -> linear out (4352 -> d_model).
//!   * 24 conformer blocks: macaron FF (x + 0.5*FF(LN)), rel-pos MHSA (full
//!     residual), conv module (pw1 -> GLU -> dw -> LN -> SiLU -> pw2, full
//!     residual), macaron FF2, final per-block LN.
//!   * chunked_limited attention mask: chunk_size = att_right + 1, left_chunks
//!     = att_left / chunk_size; band [k_min, k_max).
//!   * pos_emb computed host-side (sin/cos, div = 10000^(-2k/d)).
//!   * optional prompt MLP on the encoder output (multilingual).

use anyhow::{Context, Result};
use rayon::prelude::*;

use super::{
    config::EncoderConfig,
    timing,
    weights::{
        Conv1dDw, Conv2d, LayerNorm, Lin, load_conv1d_dw, load_conv2d,
        load_lin, load_ln,
    },
};
use crate::{
    gguf::Gguf,
    math::{f32_bytes, transpose, transpose_into},
};

pub const LN_EPS: f32 = 1e-5;

#[derive(Debug)]
pub struct PreEncode {
    pub conv0: Conv2d,
    pub conv2: Conv2d,
    pub conv3: Conv2d,
    pub conv5: Conv2d,
    pub conv6: Conv2d,
    pub out: Lin,
}

#[derive(Debug)]
pub struct PromptMlp {
    pub mlp0: Lin,
    pub mlp2: Lin,
}

#[derive(Debug)]
pub struct Block {
    pub norm_ff1: LayerNorm,
    pub ff1_lin1: Lin,
    pub ff1_lin2: Lin,
    pub norm_att: LayerNorm,
    pub attn_q: Lin,
    pub attn_k: Lin,
    pub attn_v: Lin,
    pub attn_pos: Lin,
    pub attn_out: Lin,
    /// [n_heads * head_dim], row per head.
    pub pos_u: Vec<f32>,
    pub pos_v: Vec<f32>,
    pub norm_conv: LayerNorm,
    pub pw1: Lin,
    pub dw: Conv1dDw,
    /// LayerNorm affine from the conv.batch_norm tensors
    /// (conv_norm = layer_norm).
    pub conv_ln: LayerNorm,
    pub pw2: Lin,
    pub norm_ff2: LayerNorm,
    pub ff2_lin1: Lin,
    pub ff2_lin2: Lin,
    pub norm_out: LayerNorm,
}

#[derive(Debug)]
pub struct Encoder {
    pub cfg: EncoderConfig,
    pub pre_encode: PreEncode,
    pub blocks: Vec<Block>,
    pub prompt: Option<PromptMlp>,
    /// Shared Q8 dequantization scratch (largest single matrix, not the
    /// whole model), reused across all layers.
    pub scratch: Vec<f32>,
}

impl PreEncode {
    fn load(gguf: &Gguf, channels: usize, _subsampling: usize) -> Result<Self> {
        let p = "encoder.pre_encode.conv";
        // Causal pre_encode: (k-1, s-1) pads on both axes, conv p=0.
        // Each of the three stride-2 convs halves time and freq
        // (total subsampling 8).
        let pad = (2, 1);
        let stride = 2;
        let conv0 = load_conv2d(
            gguf,
            &format!("{p}.0"),
            1,
            channels,
            3,
            3,
            pad,
            pad,
            stride,
            1,
        )?;
        let conv2 = load_conv2d(
            gguf,
            &format!("{p}.2"),
            channels,
            channels,
            3,
            3,
            pad,
            pad,
            stride,
            channels,
        )?;
        let conv3 = load_conv2d(
            gguf,
            &format!("{p}.3"),
            channels,
            channels,
            1,
            1,
            (0, 0),
            (0, 0),
            1,
            1,
        )?;
        let conv5 = load_conv2d(
            gguf,
            &format!("{p}.5"),
            channels,
            channels,
            3,
            3,
            pad,
            pad,
            stride,
            channels,
        )?;
        let conv6 = load_conv2d(
            gguf,
            &format!("{p}.6"),
            channels,
            channels,
            1,
            1,
            (0, 0),
            (0, 0),
            1,
            1,
        )?;
        // out: [channels * 17, d_model]; 17 = floor((128+2+1-3)/2)+1 thrice.
        let feat_in = 17usize * channels;
        let out = load_lin(gguf, "encoder.pre_encode.out", feat_in, 0)?;
        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }
}

impl Block {
    fn load(gguf: &Gguf, i: usize, cfg: &EncoderConfig) -> Result<Self> {
        let p = format!("encoder.layers.{i}");
        let d = cfg.d_model;
        let (d_ff, conv_k) = (cfg.d_ff, cfg.conv_kernel);
        let conv_pad = (cfg.conv_context_left, cfg.conv_context_right);
        let norm_ff1 =
            load_ln(gguf, &format!("{p}.norm_feed_forward1"), d, LN_EPS)?;
        let ff1_lin1 =
            load_lin(gguf, &format!("{p}.feed_forward1.linear1"), d, d_ff)?;
        let ff1_lin2 =
            load_lin(gguf, &format!("{p}.feed_forward1.linear2"), d_ff, d)?;
        let norm_att = load_ln(gguf, &format!("{p}.norm_self_att"), d, LN_EPS)?;
        let attn_q = load_lin(gguf, &format!("{p}.self_attn.linear_q"), d, d)?;
        let attn_k = load_lin(gguf, &format!("{p}.self_attn.linear_k"), d, d)?;
        let attn_v = load_lin(gguf, &format!("{p}.self_attn.linear_v"), d, d)?;
        let attn_pos =
            load_lin(gguf, &format!("{p}.self_attn.linear_pos"), d, d)?;
        let attn_out =
            load_lin(gguf, &format!("{p}.self_attn.linear_out"), d, d)?;
        let pos_u = gguf.read_f32(
            gguf.tensor(&format!("{p}.self_attn.pos_bias_u"))
                .with_context(|| {
                    format!("GGUF tensor {p}.self_attn.pos_bias_u not found")
                })?,
        )?;
        let pos_v = gguf.read_f32(
            gguf.tensor(&format!("{p}.self_attn.pos_bias_v"))
                .with_context(|| {
                    format!("GGUF tensor {p}.self_attn.pos_bias_v not found")
                })?,
        )?;
        let norm_conv = load_ln(gguf, &format!("{p}.norm_conv"), d, LN_EPS)?;
        let pw1 =
            load_lin(gguf, &format!("{p}.conv.pointwise_conv1"), d, 2 * d)?;
        let dw = load_conv1d_dw(
            gguf,
            &format!("{p}.conv.depthwise_conv"),
            d,
            conv_k,
            conv_pad.0,
            conv_pad.1,
        )?;
        let conv_ln =
            load_ln(gguf, &format!("{p}.conv.batch_norm"), d, LN_EPS)?;
        let pw2 = load_lin(gguf, &format!("{p}.conv.pointwise_conv2"), d, d)?;
        let norm_ff2 =
            load_ln(gguf, &format!("{p}.norm_feed_forward2"), d, LN_EPS)?;
        let ff2_lin1 =
            load_lin(gguf, &format!("{p}.feed_forward2.linear1"), d, d_ff)?;
        let ff2_lin2 =
            load_lin(gguf, &format!("{p}.feed_forward2.linear2"), d_ff, d)?;
        let norm_out = load_ln(gguf, &format!("{p}.norm_out"), d, LN_EPS)?;
        Ok(Self {
            norm_ff1,
            ff1_lin1,
            ff1_lin2,
            norm_att,
            attn_q,
            attn_k,
            attn_v,
            attn_pos,
            attn_out,
            pos_u,
            pos_v,
            norm_conv,
            pw1,
            dw,
            conv_ln,
            pw2,
            norm_ff2,
            ff2_lin1,
            ff2_lin2,
            norm_out,
        })
    }
}

impl Encoder {
    pub fn load(gguf: &Gguf, cfg: EncoderConfig) -> Result<Self> {
        let pre_encode =
            PreEncode::load(gguf, cfg.conv_channels, cfg.subsampling_factor)?;
        if let (Some(dir), Some(q)) = (timing::dump_dir(), &pre_encode.out.q) {
            let mut s = Vec::new();
            q.to_f32(&mut s);
            std::fs::write(dir.join("w_pre_out.bin"), f32_bytes(&s)).ok();
        }
        let blocks = (0..cfg.n_layers)
            .map(|i| Block::load(gguf, i, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let prompt = if gguf.tensor("prompt_kernel.0.weight").is_some() {
            Some(PromptMlp {
                mlp0: load_lin(
                    gguf,
                    "prompt_kernel.0",
                    cfg.d_model + cfg.num_prompts,
                    0,
                )?,
                mlp2: load_lin(gguf, "prompt_kernel.2", 2048, cfg.d_model)?,
            })
        } else {
            None
        };
        Ok(Self {
            cfg,
            pre_encode,
            blocks,
            prompt,
            scratch: Vec::new(),
        })
    }

    /// Host-side sinusoidal pos_emb, [pos_len, d_model] time-major rows.
    /// pos_len = 2*T_enc - 1, zero at row T_enc - 1.
    pub fn pos_emb(&self, t_enc: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        let pos_len = 2 * t_enc - 1;
        let zero_index = t_enc - 1;
        let ln10000 = 10000.0f32.ln();
        let mut pe = vec![0.0f32; pos_len * d];
        for i in 0..pos_len {
            let pos = (zero_index as isize - i as isize) as f32;
            let row = &mut pe[i * d..(i + 1) * d];
            for k in 0..d / 2 {
                let div = (-2.0 * (k as f32) * ln10000 / d as f32).exp();
                row[2 * k] = (pos * div).sin();
                row[2 * k + 1] = (pos * div).cos();
            }
        }
        pe
    }

    /// Full offline encode: mel [t_mel, n_mels] time-major in, encoder
    /// output [t_enc, d_model] time-major out. `prompt_id` selects the
    /// one-hot for the prompt MLP (None skips the prompt MLP).
    pub fn encode(
        &mut self,
        mel: &[f32],
        t_mel: usize,
        prompt_id: Option<u32>,
        out: &mut Vec<f32>,
    ) -> Result<usize> {
        let d = self.cfg.d_model;
        let n_mels = self.cfg.feat_in;

        // ---- pre_encode (causal) ----
        timing::tick("pre_encode");
        let mut x = self.pre_encode_forward(mel, t_mel, n_mels);

        let t_enc = x.len() / d;

        if let Some(dir) = timing::dump_dir() {
            std::fs::write(dir.join("enc_pre.bin"), f32_bytes(&x)).ok();
        }

        // ---- xscaling (NeMo RelPositionalEncoding; false for this model) ----
        if self.cfg.xscaling {
            let s = (d as f32).sqrt();
            for v in &mut x {
                *v *= s;
            }
        }

        // ---- conformer blocks ----
        let pe = self.pos_emb(t_enc);
        timing::tick("blocks");
        let dump_dir = timing::dump_dir();
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let dump_b0 = dump_dir.is_some() && i == 0;
            x = Self::block_forward(
                &self.cfg,
                b,
                &x,
                t_enc,
                &pe,
                &mut self.scratch,
                dump_b0,
                dump_dir,
            );
            if let Some(dir) = &dump_dir {
                std::fs::write(
                    dir.join(format!("enc_b{i}.bin")),
                    f32_bytes(&x),
                )
                .ok();
            }
        }

        // ---- prompt MLP ----
        self.apply_prompt(&mut x, t_enc, prompt_id);

        *out = x;
        timing::report();
        Ok(t_enc)
    }

    /// Apply the multilingual prompt MLP to encoder output `x` in place.
    /// `t_enc` is the number of encoder frames; with `prompt_id == None`
    /// (or out of range) the output is left unchanged. Used by both the
    /// CPU `encode` and the offline GPU path so the two stay identical.
    pub fn apply_prompt(
        &mut self,
        x: &mut Vec<f32>,
        t_enc: usize,
        prompt_id: Option<u32>,
    ) {
        let d = self.cfg.d_model;
        if let Some(mlp) = &mut self.prompt {
            let num_prompts = self.cfg.num_prompts;
            if let Some(pid) = prompt_id {
                if (pid as usize) < num_prompts {
                    let cat_in = d + num_prompts;
                    let mut cat = vec![0.0f32; cat_in * t_enc];
                    for t in 0..t_enc {
                        cat[t * cat_in..t * cat_in + d]
                            .copy_from_slice(&x[t * d..(t + 1) * d]);
                        cat[t * cat_in + d + pid as usize] = 1.0;
                    }
                    let xt = transpose(&cat, t_enc, cat_in);
                    let mut h = Vec::new();
                    mlp.mlp0.forward_t(&mut self.scratch, &xt, t_enc, &mut h);
                    crate::simd_kernel::relu_into(&mut h);
                    let mut y = Vec::new();
                    mlp.mlp2.forward_t(&mut self.scratch, &h, t_enc, &mut y);
                    *x = transpose(&y, d, t_enc);
                }
            }
        }
    }

    pub fn pre_encode_forward(
        &mut self,
        mel: &[f32],
        t_mel: usize,
        n_mels: usize,
    ) -> Vec<f32> {
        let st = std::env::var_os("STEALCODE_TIMING").is_some();
        let mut t0 = std::time::Instant::now();
        let pre_tick = |name: &str, t0: &mut std::time::Instant| {
            if st {
                eprintln!("[senc] {name} {:?}", t0.elapsed());
            }
            *t0 = std::time::Instant::now();
        };
        let pe = &mut self.pre_encode;
        let dump_dir = timing::dump_dir();
        let mut y = Vec::new();
        let mut x = mel.to_vec();
        let (mut t, mut f) = (t_mel, n_mels);
        let relu = |v: &mut Vec<f32>| crate::simd_kernel::relu_into(v);
        // conv0: 1 -> C, k=3 s=2, causal pad.
        pe.conv0.forward(&x, t, f, &mut y);
        relu(&mut y);
        x = std::mem::take(&mut y);
        (t, f) = (pe.conv0.t_out(t), pe.conv0.f_out(f));
        pre_tick("pre_c0", &mut t0);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_c0.bin"), f32_bytes(&x)).ok();
        }
        // conv2 (dw) -> conv3 (pw) -> relu
        pe.conv2.forward(&x, t, f, &mut y);
        x = std::mem::take(&mut y);
        (t, f) = (pe.conv2.t_out(t), pe.conv2.f_out(f));
        pre_tick("pre_c2", &mut t0);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_c2.bin"), f32_bytes(&x)).ok();
        }
        pe.conv3.forward(&x, t, f, &mut y);
        relu(&mut y);
        x = std::mem::take(&mut y);
        pre_tick("pre_c3", &mut t0);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_c3.bin"), f32_bytes(&x)).ok();
        }
        // conv5 (dw) -> conv6 (pw) -> relu
        pe.conv5.forward(&x, t, f, &mut y);
        x = std::mem::take(&mut y);
        (t, f) = (pe.conv5.t_out(t), pe.conv5.f_out(f));
        pre_tick("pre_c5", &mut t0);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_c5.bin"), f32_bytes(&x)).ok();
        }
        pe.conv6.forward(&x, t, f, &mut y);
        relu(&mut y);
        x = std::mem::take(&mut y);
        pre_tick("pre_c6", &mut t0);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_c6.bin"), f32_bytes(&x)).ok();
        }
        // flatten [t, f, c] -> [t, c*f] (c slow, f fast), linear out.
        let d = self.cfg.d_model;
        let flat = f * self.cfg.conv_channels;
        let mut flat_x = vec![0.0f32; t * flat];
        for tt in 0..t {
            for c in 0..self.cfg.conv_channels {
                for ff in 0..f {
                    flat_x[tt * flat + c * f + ff] =
                        x[(tt * f + ff) * self.cfg.conv_channels + c];
                }
            }
        }
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_flat.bin"), f32_bytes(&flat_x)).ok();
        }
        let xt = transpose(&flat_x, t, flat);
        let mut y_t = Vec::new();
        pe.out.forward_t(&mut self.scratch, &xt, t, &mut y_t);
        if let Some(dir) = &dump_dir {
            std::fs::write(dir.join("enc_lin_t.bin"), f32_bytes(&y_t)).ok();
        }
        transpose_into(&y_t, d, t, &mut x);
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn macaron_ff(
        scratch: &mut Vec<f32>,
        lin1: &Lin,
        lin2: &Lin,
        norm: &LayerNorm,
        x: &[f32],
        t: usize,
        d: usize,
        y: &mut Vec<f32>,
    ) {
        let mut ln = vec![0.0f32; t * d];
        for tt in 0..t {
            norm.forward(
                &x[tt * d..(tt + 1) * d],
                &mut ln[tt * d..(tt + 1) * d],
            );
        }
        let xt = transpose(&ln, t, d);
        let mut h = Vec::new();
        lin1.forward_t(scratch, &xt, t, &mut h); // [d_ff, t]
        crate::simd_kernel::silu_into(&mut h);
        // h is already [d_ff, t] (out-row-major) - the layout forward_t wants.
        let mut f = Vec::new();
        lin2.forward_t(scratch, &h, t, &mut f); // [d, t]
        let f = transpose(&f, lin2.out, t); // [t, d]
        // y = x + 0.5 * f
        y.resize(t * d, 0.0);
        for i in 0..t * d {
            y[i] = x[i] + 0.5 * f[i];
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn block_forward(
        cfg: &EncoderConfig,
        b: &mut Block,
        x: &[f32],
        t: usize,
        pe: &[f32],
        scratch: &mut Vec<f32>,
        dump: bool,
        dump_dir: Option<&std::path::PathBuf>,
    ) -> Vec<f32> {
        let d = cfg.d_model;
        let n_heads = cfg.n_heads;
        let head_dim = d / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let left = cfg.att_context_left;
        let right = cfg.att_context_right;
        let dmp = |name: &str, v: &[f32]| {
            if dump {
                if let Some(dir) = dump_dir {
                    std::fs::write(dir.join(name), f32_bytes(v)).ok();
                }
            }
        };

        // ---- macaron FF1 ----
        let mut y = Vec::new();
        Self::macaron_ff(
            scratch,
            &b.ff1_lin1,
            &b.ff1_lin2,
            &b.norm_ff1,
            x,
            t,
            d,
            &mut y,
        );
        timing::tick("ff1");
        dmp("enc_ff1.bin", &y);

        // ---- rel-pos MHSA ----
        let mut ln = vec![0.0f32; t * d];
        for tt in 0..t {
            b.norm_att.forward(
                &y[tt * d..(tt + 1) * d],
                &mut ln[tt * d..(tt + 1) * d],
            );
        }
        dmp("enc_ln_attn.bin", &ln);
        let xt = transpose(&ln, t, d);
        let mut q = Vec::new();
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut p = Vec::new();
        b.attn_q.forward_t(scratch, &xt, t, &mut q); // [d, t]
        b.attn_k.forward_t(scratch, &xt, t, &mut k);
        b.attn_v.forward_t(scratch, &xt, t, &mut v);
        let pet = transpose(pe, 2 * t - 1, d); // pe rows = 2t-1 -> [d, 2t-1]
        b.attn_pos.forward_t(scratch, &pet, 2 * t - 1, &mut p);
        timing::tick("qkv");
        let q = transpose(&q, d, t);
        let k = transpose(&k, d, t);
        let v = transpose(&v, d, t);
        let p = transpose(&p, d, 2 * t - 1);

        // scores[q,k,h] = scale*(q_u.k + q_v.p[k-q+T-1]); pairs outside the
        // attention band [k_min, k_max) are skipped (mask is -inf there).
        let chunk_size = right + 1;
        let left_chunks = left / chunk_size;
        let band = |qq: usize| {
            let q_chunk = qq / chunk_size;
            let k_min = q_chunk.saturating_sub(left_chunks) * chunk_size;
            let k_max = ((q_chunk + 1) * chunk_size).min(t);
            (k_min, k_max)
        };
        let mut scores = vec![0.0f32; t * t * n_heads];
        let compute_scores = |qq: usize, row: &mut [f32]| {
            let (k_min, k_max) = band(qq);
            for h in 0..n_heads {
                let hd = h * head_dim;
                let uh = &b.pos_u[hd..hd + head_dim];
                let vh = &b.pos_v[hd..hd + head_dim];
                let qu = &q[qq * d + hd..qq * d + hd + head_dim];
                let qv = &q[qq * d + hd..qq * d + hd + head_dim];
                for kk in k_min..k_max {
                    let kk_d = &k[kk * d + hd..kk * d + hd + head_dim];
                    let pos_row = &p[(kk + t - qq - 1) * d + hd
                        ..(kk + t - qq - 1) * d + hd + head_dim];
                    row[kk * n_heads + h] = crate::simd_kernel::score_dot(
                        qu, uh, qv, vh, kk_d, pos_row, scale,
                    );
                }
            }
        };
        if t >= 8 {
            scores
                .par_chunks_mut(t * n_heads)
                .enumerate()
                .for_each(|(qq, row)| compute_scores(qq, row));
        } else {
            for qq in 0..t {
                compute_scores(qq, &mut scores[qq * t * n_heads..]);
            }
        }
        timing::tick("scores");
        dmp("enc_scores.bin", &scores);
        // softmax over k + weighted sum of v within the band. Each row runs on
        // its own disjoint slices, so the in-place exp below is rayon-safe.
        let mut attn_out = vec![0.0f32; t * d];
        let softmax_row = |qq: usize, srow: &mut [f32], row: &mut [f32]| {
            let (k_min, k_max) = band(qq);
            for h in 0..n_heads {
                let hd = h * head_dim;
                crate::simd_kernel::softmax_v(
                    srow,
                    k_min,
                    k_max,
                    n_heads,
                    h,
                    |i| &v[i * d + hd..i * d + hd + head_dim],
                    head_dim,
                    &mut row[hd..hd + head_dim],
                );
            }
        };
        if t >= 8 {
            scores
                .par_chunks_mut(t * n_heads)
                .zip(attn_out.par_chunks_mut(d))
                .enumerate()
                .for_each(|(qq, (srow, row))| softmax_row(qq, srow, row));
        } else {
            for qq in 0..t {
                softmax_row(
                    qq,
                    &mut scores[qq * t * n_heads..(qq + 1) * t * n_heads],
                    &mut attn_out[qq * d..(qq + 1) * d],
                );
            }
        }
        timing::tick("softmax_v");
        let at = transpose(&attn_out, t, d);
        let mut o = Vec::new();
        b.attn_out.forward_t(scratch, &at, t, &mut o);
        let o = transpose(&o, d, t);
        for i in 0..t * d {
            y[i] += o[i];
        }
        timing::tick("attn_out");
        dmp("enc_attn_res.bin", &y);

        // ---- conv module ----
        for tt in 0..t {
            b.norm_conv.forward(
                &y[tt * d..(tt + 1) * d],
                &mut ln[tt * d..(tt + 1) * d],
            );
        }
        let xt = transpose(&ln, t, d);
        let mut h = Vec::new();
        b.pw1.forward_t(scratch, &xt, t, &mut h);
        let h = transpose(&h, 2 * d, t);
        // GLU: gate * sigmoid(value)
        let mut glu = vec![0.0f32; t * d];
        crate::simd_kernel::glu_from(&h, d, &mut glu);
        dmp("enc_glu.bin", &glu);
        // depthwise conv with left pad (conv_context_left, 0)
        let mut conv = Vec::new();
        b.dw.forward(&glu, t, &mut conv);
        dmp("enc_dw.bin", &conv);
        // LN over channels (affine = batch_norm tensors)
        for tt in 0..conv.len() / d {
            b.conv_ln.forward(
                &conv[tt * d..(tt + 1) * d],
                &mut ln[tt * d..(tt + 1) * d],
            );
            let ln_slice = &mut ln[tt * d..(tt + 1) * d];
            crate::simd_kernel::silu_into(ln_slice);
            conv[tt * d..(tt + 1) * d].copy_from_slice(ln_slice);
        }
        dmp("enc_convln.bin", &conv);
        let ct = transpose(&conv, t, d);
        let mut o2 = Vec::new();
        b.pw2.forward_t(scratch, &ct, t, &mut o2);
        let o2 = transpose(&o2, d, t);
        for i in 0..t * d {
            y[i] += o2[i];
        }
        timing::tick("conv_mod");
        dmp("enc_conv_res.bin", &y);

        // ---- macaron FF2 ----
        let mut y2 = Vec::new();
        Self::macaron_ff(
            scratch,
            &b.ff2_lin1,
            &b.ff2_lin2,
            &b.norm_ff2,
            &y,
            t,
            d,
            &mut y2,
        );

        // ---- final per-block LN ----
        let mut out = vec![0.0f32; t * d];
        for tt in 0..t {
            b.norm_out.forward(
                &y2[tt * d..(tt + 1) * d],
                &mut out[tt * d..(tt + 1) * d],
            );
        }
        timing::tick("ln_out");
        out
    }
}
