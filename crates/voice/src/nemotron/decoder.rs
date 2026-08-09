//! RNN-T decoder: predictor (2-layer LSTM) + joint network + greedy
//! search. Mirrors the C++ reference `decode_rnnt_greedy`:
//!   * gates = Wx@x + Wh@h + (b_ih + b_hh), order [i, f, g, o], c' = f*c + i*g,
//!     h' = o*tanh(c').
//!   * embed lookup = embed_w[last_token] (row 0 for the start state is NOT
//!     used вЂ” the C++ reference feeds zeros on last_token < 0).
//!   * joint: logits = out_w @ relu(enc_proj[step] + pred_proj) + out_b (joint
//!     activation = relu for this model).
//!   * blank: step += 1 (predictor state unchanged); token: emit, last_token =
//!     token, swap state. max_symbols_per_step caps consecutive tokens (then
//!     step += 1).

use anyhow::{Context, Result};
use tracing::debug;

use super::{
    config::RnntConfig,
    gguf::Gguf,
    weights::{Lin, load_lin},
};

#[derive(Debug)]
pub struct LstmLayer {
    pub ih: Lin,
    pub hh: Lin,
    /// ih_bias + hh_bias folded.
    pub bias: Vec<f32>,
    pub hidden: usize,
}

#[derive(Debug)]
pub struct Predictor {
    pub embed: Vec<f32>,
    pub layers: Vec<LstmLayer>,
    pub hidden: usize,
}

#[derive(Debug)]
pub struct Joint {
    pub enc: Lin,
    pub pred: Lin,
    pub out: Lin,
    pub d_enc: usize,
    pub joint_h: usize,
    pub n_cls: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub id: u32,
    pub p: f32,
    /// Encoder frame index at emission (step_at_emit).
    pub step: usize,
}

#[derive(Debug)]
pub struct GreedyDecoder {
    pub predictor: Predictor,
    pub joint: Joint,
    pub blank_id: u32,
    pub max_symbols: usize,
}

impl LstmLayer {
    fn load(gguf: &Gguf, i: usize, hidden: usize) -> Result<Self> {
        let p = format!("decoder.prediction.dec_rnn.lstm.ih_l{i}");
        let h = format!("decoder.prediction.dec_rnn.lstm.hh_l{i}");
        let ih = load_lin(gguf, &format!("{p}"), hidden, 4 * hidden)?;
        let hh = load_lin(gguf, &format!("{h}"), hidden, 4 * hidden)?;
        let mut bias = Vec::with_capacity(4 * hidden);
        {
            let m = gguf
                .tensor(&format!("{p}.bias"))
                .with_context(|| format!("GGUF tensor {p}.bias not found"))?;
            let a = gguf.read_f32(m)?;
            let m = gguf
                .tensor(&format!("{h}.bias"))
                .with_context(|| format!("GGUF tensor {h}.bias not found"))?;
            let b = gguf.read_f32(m)?;
            if a.len() != 4 * hidden || b.len() != 4 * hidden {
                anyhow::bail!("{p}.bias: unexpected size");
            }
            for k in 0..4 * hidden {
                bias.push(a[k] + b[k]);
            }
        }
        Ok(Self {
            ih,
            hh,
            bias,
            hidden,
        })
    }
}

impl Predictor {
    fn load(gguf: &Gguf, cfg: &RnntConfig) -> Result<Self> {
        let hidden = cfg.pred_hidden;
        let embed = {
            let m = gguf.tensor("decoder.prediction.embed.weight").context(
                "GGUF tensor decoder.prediction.embed.weight not found",
            )?;
            let v = gguf.read_f32(m)?;
            if v.len() != cfg.vocab_size * hidden {
                anyhow::bail!("embed size {} != vocab*hidden", v.len());
            }
            v
        };
        let layers = (0..cfg.pred_n_layers)
            .map(|i| LstmLayer::load(gguf, i, hidden))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed,
            layers,
            hidden,
        })
    }

    /// One LSTM step. `last_token < 0` feeds zeros (start state).
    pub fn step(
        &mut self,
        last_token: i32,
        h: &[Vec<f32>],
        c: &[Vec<f32>],
        nh: &mut [Vec<f32>],
        nc: &mut [Vec<f32>],
        x: &mut [f32],
    ) {
        let hdim = self.hidden;
        if last_token < 0 {
            x.fill(0.0);
        } else {
            let base = last_token as usize * hdim;
            x.copy_from_slice(&self.embed[base..base + hdim]);
        }
        let mut gates = vec![0.0f32; 4 * hdim];
        let mut x_in = x.to_vec();
        for l in 0..self.layers.len() {
            let layer = &mut self.layers[l];
            layer.ih.matvec_nb(&x_in, &mut gates);
            let mut g2 = vec![0.0f32; 4 * hdim];
            layer.hh.matvec_nb(&h[l], &mut g2);
            for k in 0..4 * hdim {
                gates[k] += g2[k] + layer.bias[k];
            }
            let (i, f, g, o) = (
                &gates[0..hdim],
                &gates[hdim..2 * hdim],
                &gates[2 * hdim..3 * hdim],
                &gates[3 * hdim..4 * hdim],
            );
            for k in 0..hdim {
                let ig = sigmoid(i[k]);
                let fg = sigmoid(f[k]);
                let gg = g[k].tanh();
                let og = sigmoid(o[k]);
                let cn = fg * c[l][k] + ig * gg;
                nc[l][k] = cn;
                nh[l][k] = og * cn.tanh();
            }
            if l == 0 {
                if let Some(dir) = std::env::var_os("STEALCODE_DUMP_DIR")
                    .map(std::path::PathBuf::from)
                {
                    let mut v: Vec<f32> = Vec::with_capacity(4 * hdim);
                    for k in 0..hdim {
                        v.push(sigmoid(i[k]));
                        v.push(g[k].tanh());
                        v.push(sigmoid(o[k]));
                        v.push(nh[0][k]);
                    }
                    std::fs::write(dir.join("acts0.bin"), bytemuck_enc(&v))
                        .ok();
                    std::fs::write(
                        dir.join("real_gates0.bin"),
                        bytemuck_enc(&gates),
                    )
                    .ok();
                }
            }
            x_in = nh[l].clone();
        }
    }
}

impl Joint {
    fn load(gguf: &Gguf, cfg: &RnntConfig) -> Result<Self> {
        // joint.enc.weight gguf dims (1024, 640) -> rows = 640, len = 1024.
        let enc = load_lin(gguf, "joint.enc", 1024, 0)?;
        let pred = load_lin(gguf, "joint.pred", cfg.joint_dim, cfg.joint_dim)?;
        let out = load_lin(gguf, "joint.joint_net.2", cfg.joint_dim, 0)?;
        Ok(Self {
            enc,
            pred,
            out,
            d_enc: 1024,
            joint_h: cfg.joint_dim,
            n_cls: cfg.vocab_size,
        })
    }
}

impl GreedyDecoder {
    pub fn load(gguf: &Gguf, cfg: &RnntConfig) -> Result<Self> {
        Ok(Self {
            predictor: Predictor::load(gguf, cfg)?,
            joint: Joint::load(gguf, cfg)?,
            blank_id: cfg.blank_id as u32,
            max_symbols: cfg.max_symbols_per_step,
        })
    }

    /// Greedy decode over encoder output `enc` (time-major
    /// [t_enc, d_enc]).
    pub fn decode(&mut self, enc: &[f32], t_enc: usize) -> Result<Vec<Token>> {
        let dump = std::env::var_os("STEALCODE_DUMP_DIR")
            .map(std::path::PathBuf::from);
        let d_enc = self.joint.d_enc;
        let joint_h = self.joint.joint_h;
        let n_cls = self.joint.n_cls;
        let blank = self.blank_id as usize;

        // Precompute enc projections: enc_proj[t] = enc_w @ enc[t] + enc_b.
        let mut enc_proj = vec![0.0f32; t_enc * joint_h];
        let mut scratch = Vec::new();
        {
            let xt = transpose(enc, t_enc, d_enc);
            let mut y = Vec::new();
            self.joint.enc.forward_t(&mut scratch, &xt, t_enc, &mut y);
            if y.iter().any(|v| v.is_nan()) {
                // Debug the first matmul output column.
                let w0: Vec<f32> = if let Some(q) = &self.joint.enc.q {
                    let mut s = Vec::new();
                    q.to_f32(&mut s);
                    s
                } else {
                    self.joint.enc.f.clone().unwrap_or_default()
                };
                debug!(
                    "joint.enc deq: len={} maxabs={} first={:?}",
                    w0.len(),
                    w0.iter().fold(0.0f32, |a, &b| a.max(b.abs())),
                    &w0[..4]
                );
                debug!(
                    "x_t[..4]={:?} x_t maxabs={}",
                    &xt[..4],
                    xt.iter().fold(0.0f32, |a, &b| a.max(b.abs()))
                );
                debug!(
                    "y[..4]={:?} y nan={}",
                    &y[..4],
                    y.iter().filter(|v| v.is_nan()).count()
                );
                // Manual dot for y[0] (col 0): sum w0[0][i]*xt[i]
                let mut acc = 0.0f64;
                for i in 0..1024 {
                    acc += w0[i] as f64 * xt[i] as f64;
                }
                debug!("manual y[0]={acc} (via f64)");
            }
            for t in 0..t_enc {
                for j in 0..joint_h {
                    enc_proj[t * joint_h + j] = y[j * t_enc + t];
                }
            }
        }
        if let Some(dir) = &dump {
            std::fs::write(dir.join("enc.bin"), bytemuck_enc(enc)).ok();
            std::fs::write(dir.join("enc_proj.bin"), bytemuck_enc(&enc_proj))
                .ok();
        }

        let hdim = self.predictor.hidden;
        let n_layers = self.predictor.layers.len();
        let mut h = vec![vec![0.0f32; hdim]; n_layers];
        let mut c = vec![vec![0.0f32; hdim]; n_layers];
        let mut nh = vec![vec![0.0f32; hdim]; n_layers];
        let mut nc = vec![vec![0.0f32; hdim]; n_layers];
        let mut embed_x = vec![0.0f32; hdim];

        let mut out_tokens = Vec::new();
        let mut last_token: i32 = -1;
        let mut step = 0usize;
        let mut new_symbols = 0usize;
        let mut iter = 0usize;
        let max_iters = 16 * t_enc + 1024;
        let mut predictor_dirty = true;

        let mut pred_proj = vec![0.0f32; joint_h];
        let mut summed = vec![0.0f32; joint_h];
        let mut logits = vec![0.0f32; n_cls];
        let mut probs = vec![0.0f32; n_cls];

        let e0 = &enc_proj[0..joint_h];
        debug!(
            "enc_proj[0][..4]={:?} nan={}",
            &e0[..4],
            e0.iter().filter(|v| v.is_nan()).count()
        );

        while step < t_enc && iter < max_iters {
            iter += 1;
            if predictor_dirty {
                self.predictor.step(
                    last_token,
                    &h,
                    &c,
                    &mut nh,
                    &mut nc,
                    &mut embed_x,
                );
                predictor_dirty = false;
            }
            // decoder_out = nh last layer.
            {
                let dec = &nh[n_layers - 1];
                self.joint.pred.matvec(dec, &mut pred_proj);
            }
            if iter <= 2 {
                debug!(
                    "iter {iter}: h[..3]={:?} h_nan={} pred_proj[..3]={:?} pp_nan={}",
                    &nh[n_layers - 1][..3],
                    nh[n_layers - 1].iter().filter(|v| v.is_nan()).count(),
                    &pred_proj[..3],
                    pred_proj.iter().filter(|v| v.is_nan()).count()
                );
            }
            let e = &enc_proj[step * joint_h..(step + 1) * joint_h];
            for j in 0..joint_h {
                summed[j] = (e[j] + pred_proj[j]).max(0.0); // relu
            }
            self.joint.out.matvec(&summed, &mut logits);

            let mut best = 0usize;
            let mut best_v = logits[0];
            for i in 1..n_cls {
                if logits[i] > best_v {
                    best_v = logits[i];
                    best = i;
                }
            }
            if iter == 1 {
                if let Some(dir) = &dump {
                    std::fs::write(dir.join("h0_l0.bin"), bytemuck_enc(&nh[0]))
                        .ok();
                    std::fs::write(dir.join("h0_l1.bin"), bytemuck_enc(&nh[1]))
                        .ok();
                    std::fs::write(
                        dir.join("pred_proj0.bin"),
                        bytemuck_enc(&pred_proj),
                    )
                    .ok();
                    std::fs::write(
                        dir.join("logits0.bin"),
                        bytemuck_enc(&logits),
                    )
                    .ok();
                    let g0 = self.predictor.layers[0].bias.clone();
                    std::fs::write(dir.join("gates0.bin"), bytemuck_enc(&g0))
                        .ok();
                    debug!("dump: best={best} best_v={best_v}");
                }
            }

            if best == blank {
                step += 1;
                new_symbols = 0;
            } else {
                let p = token_confidence(&logits, n_cls, &mut probs);
                out_tokens.push(Token {
                    id: best as u32,
                    p,
                    step,
                });
                last_token = best as i32;
                std::mem::swap(&mut h, &mut nh);
                std::mem::swap(&mut c, &mut nc);
                predictor_dirty = true;
                new_symbols += 1;
                if self.max_symbols > 0 && new_symbols >= self.max_symbols {
                    step += 1;
                    new_symbols = 0;
                }
            }
        }
        if iter >= max_iters {
            anyhow::bail!("rnnt decode: iteration cap ({max_iters}) hit");
        }
        debug!(
            "decode done: iters={iter} tokens={} final_step={step}",
            out_tokens.len()
        );
        Ok(out_tokens)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn bytemuck_enc(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Entropy-based confidence (matches the C++ reference):
/// p = softmax(logits), confidence = 1 - entropy/log(n_cls).
fn token_confidence(logits: &[f32], n_cls: usize, probs: &mut Vec<f32>) -> f32 {
    let mut maxv = logits[0];
    for &v in &logits[1..] {
        if v > maxv {
            maxv = v;
        }
    }
    probs.resize(n_cls, 0.0);
    let mut sum: f64 = 0.0;
    for i in 0..n_cls {
        let e = (logits[i] - maxv).exp();
        probs[i] = e;
        sum += e as f64;
    }
    let inv = (1.0 / sum) as f32;
    let mut entropy: f64 = 0.0;
    for i in 0..n_cls {
        let p = probs[i] * inv;
        entropy -= (p as f64) * ((p as f64) + 1e-10).ln();
    }
    let max_entropy = (n_cls as f64).ln();
    if max_entropy <= 0.0 {
        return 1.0;
    }
    1.0 - (entropy / max_entropy) as f32
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}
