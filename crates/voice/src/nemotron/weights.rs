//! Tensor primitives and weight loading.
//!
//! The GGUF file stores every matrix as rows (torch row-major of the
//! reversed gguf dims). Q8 matrices are kept quantized on disk and
//! dequantized on the fly during a forward pass; everything else is
//! loaded as f32.
//!
//! Quant layout note: this model's converter (NVIDIA gguf-py fork)
//! writes its "Q8_0" as 34-byte blocks `[f16 d][i8 x32]`, value = d*q,
//! NOT llama.cpp's 36-byte `[f32 d][i8 x32]`. Verified empirically:
//! every dtype-8 tensor is 34 bytes/block and dequantizes to sane
//! weights (no NaN/inf).
//!
//! Matrices are consumed as `[out, in]` row-major against activations
//! in transposed `[in, T]` layout (see `Lin::forward_t`), so a forward
//! is one sgemm (matrixmultiply, multi-threaded).

use anyhow::{Context, Result, bail};

use super::sgemm_kernel;

/// C = A @ B (row-major, `c` accumulates nothing: beta = 0) via the
/// dedicated AVX2 kernel, or matrixmultiply as fallback.
#[allow(unsafe_code)]
unsafe fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    rsa: isize,
    _csa: isize,
    b: *const f32,
    rsb: isize,
    _csb: isize,
    c: *mut f32,
    rsc: isize,
    _csc: isize,
) {
    debug_assert_eq!(rsa, k as isize);
    debug_assert_eq!(_csa, 1);
    debug_assert_eq!(rsb, n as isize);
    debug_assert_eq!(_csb, 1);
    debug_assert_eq!(rsc, n as isize);
    debug_assert_eq!(_csc, 1);
    let a_s = std::slice::from_raw_parts(a, m * k);
    let b_s = std::slice::from_raw_parts(b, k * n);
    let c_s = std::slice::from_raw_parts_mut(c, m * n);
    sgemm_kernel::gemm_into(m, k, n, a_s, b_s, c_s);
}
use rayon::prelude::*;

use super::gguf::{Gguf, f16_to_f32};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8Variant {
    /// f32 scale + i8x32, value = d*q, 36 bytes/block (llama.cpp Q8_0).
    Q8_0,
    /// f16 scale + i8x32, value = d*q, 34 bytes/block (this model's
    /// NVIDIA-converter Q8_0 layout).
    Q8F16,
}

impl Q8Variant {
    fn block_bytes(self) -> usize {
        match self {
            Q8Variant::Q8_0 => 36,
            Q8Variant::Q8F16 => 34,
        }
    }
}

/// A quantized matrix: `rows` x `row_len` values, stored as per-32
/// blocks (each row padded to a multiple of 32 elements). Weights stay
/// quantized for the whole lifetime — both `matvec` and `forward_t` run
/// the int8 vec-dot kernel directly on the stored bytes, so no f32
/// dequantization (and no multi-GB cache) is ever built.
pub struct Q8Mat {
    bytes: Vec<u8>,
    rows: usize,
    row_len: usize,
    padded_row: usize,
    variant: Q8Variant,
}

impl std::fmt::Debug for Q8Mat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Q8Mat {{ {}x{}, {:?}, {} bytes }}",
            self.rows,
            self.row_len,
            self.variant,
            self.bytes.len()
        )
    }
}

impl Q8Mat {
    pub fn new(
        bytes: Vec<u8>,
        rows: usize,
        row_len: usize,
        variant: Q8Variant,
    ) -> Result<Self> {
        let padded_row = row_len.div_ceil(32) * variant.block_bytes();
        if bytes.len() != rows * padded_row {
            bail!(
                "Q8Mat: {} bytes != {rows} rows x {padded_row} (row_len {row_len})",
                bytes.len()
            );
        }
        Ok(Self {
            bytes,
            rows,
            row_len,
            padded_row,
            variant,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Raw quantized block bytes (m rows x `padded_row`).
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Bytes per row (blocks_per_row * block_bytes).
    pub fn padded_row(&self) -> usize {
        self.padded_row
    }

    /// y[j] = sum over blocks of (d*dot(x, q_b)) + bias[j], computed as
    /// an int8 x int8 vec-dot against the block-quantized activations
    /// (n = 1 specialization of `q8_gemm`).
    pub fn matvec(&self, x: &[f32], bias: Option<&[f32]>, y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.row_len);
        debug_assert_eq!(y.len(), self.rows);
        let bb = self.variant.block_bytes();
        let qoff = if self.variant == Q8Variant::Q8F16 {
            2
        } else {
            4
        };
        sgemm_kernel::q8_gemm(
            self.rows,
            self.row_len,
            1,
            &self.bytes,
            self.padded_row,
            bb,
            qoff,
            x,
            y,
        );
        if let Some(b) = bias {
            for j in 0..self.rows {
                y[j] += b[j];
            }
        }
    }

    /// `y[m, n] = W[m, k] @ x[k, n]` for the `[inp, t]` activation layout
    /// (x row-major `[k, n]`, y row-major `[m, n]`), via the int8 kernel.
    pub fn gemm_t(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let bb = self.variant.block_bytes();
        let qoff = if self.variant == Q8Variant::Q8F16 {
            2
        } else {
            4
        };
        sgemm_kernel::q8_gemm(
            self.rows,
            self.row_len,
            t,
            &self.bytes,
            self.padded_row,
            bb,
            qoff,
            x,
            y,
        );
    }

    /// Dequantize into `out` as `[rows, row_len]` row-major f32.
    /// NOT cached — only used by debug/dump paths and the micro-benches.
    pub fn to_f32(&self, out: &mut Vec<f32>) {
        let bb = self.variant.block_bytes();
        let blocks_per_row = self.row_len.div_ceil(32);
        let qoff = if self.variant == Q8Variant::Q8F16 {
            2
        } else {
            4
        };
        out.clear();
        out.reserve(self.rows * self.row_len);
        for j in 0..self.rows {
            let base = j * self.padded_row;
            for blk in 0..blocks_per_row {
                let b = base + blk * bb;
                let d = read_block_scale(&self.bytes, b, &self.variant);
                let vals = &self.bytes[b + qoff..b + bb];
                let in_row = blk * 32;
                let keep = 32usize.min(self.row_len.saturating_sub(in_row));
                for i in 0..keep {
                    out.push(d * (vals[i] as i8 as f32));
                }
            }
        }
    }
}

fn read_block_scale(bytes: &[u8], b: usize, variant: &Q8Variant) -> f32 {
    match variant {
        Q8Variant::Q8_0 => f32::from_le_bytes([
            bytes[b],
            bytes[b + 1],
            bytes[b + 2],
            bytes[b + 3],
        ]),
        Q8Variant::Q8F16 => {
            f16_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
        }
    }
}

/// A linear layer with optional bias, stored as Q8_0/Q8F16 or f32.
pub struct Lin {
    pub q: Option<Q8Mat>,
    pub f: Option<Vec<f32>>,
    pub bias: Option<Vec<f32>>,
    pub out: usize,
    pub inp: usize,
}

impl std::fmt::Debug for Lin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lin {{ {0}->{1} }}", self.inp, self.out)
    }
}

impl Lin {
    pub fn matvec(&self, x: &[f32], y: &mut Vec<f32>) {
        y.resize(self.out, 0.0);
        if let Some(q) = &self.q {
            q.matvec(x, self.bias.as_deref(), y);
        } else if let Some(f) = &self.f {
            for j in 0..self.out {
                let row = &f[j * self.inp..(j + 1) * self.inp];
                let mut acc = 0.0f32;
                for i in 0..self.inp {
                    acc += row[i] * x[i];
                }
                y[j] = acc
                    + match &self.bias {
                        Some(b) => b[j],
                        None => 0.0,
                    };
            }
        }
    }

    /// Same as `matvec` but WITHOUT adding the layer bias (used by the
    /// RNNT predictor, which folds ih+hh biases itself).
    pub fn matvec_nb(&self, x: &[f32], y: &mut Vec<f32>) {
        y.resize(self.out, 0.0);
        if let Some(q) = &self.q {
            q.matvec(x, None, y);
        } else if let Some(f) = &self.f {
            for j in 0..self.out {
                let row = &f[j * self.inp..(j + 1) * self.inp];
                let mut acc = 0.0f32;
                for i in 0..self.inp {
                    acc += row[i] * x[i];
                }
                y[j] = acc;
            }
        }
    }

    /// `y = W @ x + b` where `x` is `[inp, t]` row-major (the transposed
    /// activation layout) and `y` comes back `[out, t]` row-major.
    /// Quantized weights are consumed directly by the int8 GEMM kernel
    /// (never dequantized to f32).
    #[allow(unsafe_code)]
    pub fn forward_t(
        &self,
        _scratch: &mut Vec<f32>,
        x_t: &[f32],
        t: usize,
        y_t: &mut Vec<f32>,
    ) {
        debug_assert_eq!(x_t.len(), self.inp * t);
        y_t.resize(self.out * t, 0.0);
        // C = A @ B: A = W [out, inp] row-major (rsa=inp, csa=1),
        // B = x [inp, t] row-major (rsb=t, csb=1),
        // C = y [out, t] row-major (rsc=t, csc=1).
        let m = self.out;
        let k = self.inp;
        let n = t;
        if let Some(q) = &self.q {
            q.gemm_t(x_t, t, y_t);
        } else if let Some(f) = &self.f {
            // SAFETY: all three buffers are writable/sized m*k, k*n, m*n
            // and disjoint; the slices outlive the call.
            unsafe {
                gemm(
                    m,
                    k,
                    n,
                    f.as_ptr(),
                    k as isize,
                    1,
                    x_t.as_ptr(),
                    n as isize,
                    1,
                    y_t.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        }
        if let Some(b) = &self.bias {
            for j in 0..self.out {
                let bj = b[j];
                for tt in 0..t {
                    y_t[j * t + tt] += bj;
                }
            }
        }
    }
}

/// 2D conv with explicit per-axis padding and groups (depthwise).
/// Input is `[t, f, c_in]` (time-major, channel-fastest — the mel and
/// activation layout used throughout this port); output is
/// `[t', f', c_out]` with `t' = t_out()`, `f' = f_out()`.
/// Kernel weights are torch `[out, in/groups, kh, kw]` row-major.
pub struct Conv2d {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out: usize,
    pub inp: usize,
    pub kh: usize,
    pub kw: usize,
    /// (left, right) padding on the time axis.
    pub pad_t: (usize, usize),
    /// (left, right) padding on the freq axis.
    pub pad_f: (usize, usize),
    pub stride_t: usize,
    pub stride_f: usize,
    pub groups: usize,
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Conv2d {{ {1}->{0} {2}x{3} g={6} pad_t={4:?} pad_f={5:?} }}",
            self.out,
            self.inp,
            self.kh,
            self.kw,
            self.pad_t,
            self.pad_f,
            self.groups
        )
    }
}

impl Conv2d {
    pub fn f_out(&self, f_in: usize) -> usize {
        (f_in + self.pad_f.0 + self.pad_f.1 - self.kw) / self.stride_f + 1
    }

    pub fn t_out(&self, t_in: usize) -> usize {
        (t_in + self.pad_t.0 + self.pad_t.1 - self.kh) / self.stride_t + 1
    }

    /// Forward over `[t_in, f_in, c_in]` input; result `[t_out, f_out,
    /// c_out]` in `out` (resized). For depthwise convs `groups == inp`
    /// (and the stored `inp` is the true in-channels).
    pub fn forward(
        &self,
        x: &[f32],
        t_in: usize,
        f_in: usize,
        out: &mut Vec<f32>,
    ) {
        let t_out = self.t_out(t_in);
        let f_out = self.f_out(f_in);
        let (c_out, c_in) = (self.out, self.inp);
        // Pointwise convs (1x1, stride 1) are dense matmuls: run them
        // through the SIMD sgemm instead of the scalar loops below
        // (these dominate the pre-encode cost).
        if self.kh == 1
            && self.kw == 1
            && self.stride_t == 1
            && self.stride_f == 1
        {
            let spatial = t_in * f_in;
            let mut xt = vec![0.0f32; c_in * spatial];
            for sp in 0..spatial {
                for c in 0..c_in {
                    xt[c * spatial + sp] = x[sp * c_in + c];
                }
            }
            let mut y_t = vec![0.0f32; c_out * spatial];
            unsafe {
                gemm(
                    c_out,
                    c_in,
                    spatial,
                    self.w.as_ptr(),
                    c_in as isize,
                    1,
                    xt.as_ptr(),
                    spatial as isize,
                    1,
                    y_t.as_mut_ptr(),
                    spatial as isize,
                    1,
                );
            }
            if !self.b.is_empty() {
                for j in 0..c_out {
                    let bj = self.b[j];
                    for sp in 0..spatial {
                        y_t[j * spatial + sp] += bj;
                    }
                }
            }
            out.resize(t_out * f_out * c_out, 0.0);
            for sp in 0..spatial {
                for j in 0..c_out {
                    out[sp * c_out + j] = y_t[j * spatial + sp];
                }
            }
            return;
        }
        out.resize(t_out * f_out * c_out, 0.0);
        let w = &self.w;
        let b = &self.b;
        let pt_l = self.pad_t.0 as isize;
        let pf_l = self.pad_f.0 as isize;
        let in_groups = c_in / self.groups;
        let spatial = t_out * f_out;
        if spatial >= 8 {
            // Parallel over spatial positions: каждый поток пишет
            // смежные c_out каналов одного пикселя (без false sharing).
            out.par_chunks_mut(c_out)
                .enumerate()
                .for_each(|(sp, slot)| {
                    let (ot, of) = (sp / f_out, sp % f_out);
                    let t0 = (ot * self.stride_t) as isize;
                    let f0 = (of * self.stride_f) as isize;
                    for oc in 0..c_out {
                        let ic0 = (oc % self.groups) * in_groups;
                        let wbase = oc * in_groups * self.kh * self.kw;
                        let mut acc = b[oc];
                        for ic in 0..in_groups {
                            let ic_abs = ic0 + ic;
                            for kt in 0..self.kh {
                                let ti = t0 + kt as isize - pt_l;
                                if ti < 0 || ti >= t_in as isize {
                                    continue;
                                }
                                for kf in 0..self.kw {
                                    let fi = f0 + kf as isize - pf_l;
                                    if fi < 0 || fi >= f_in as isize {
                                        continue;
                                    }
                                    let wi = (ic * self.kh + kt) * self.kw + kf;
                                    let xi = (ti as usize * f_in + fi as usize)
                                        * c_in
                                        + ic_abs;
                                    acc += x[xi] * w[wbase + wi];
                                }
                            }
                        }
                        slot[oc] = acc;
                    }
                });
        } else {
            for oc in 0..c_out {
                let ic0 = (oc % self.groups) * in_groups;
                let wbase = oc * in_groups * self.kh * self.kw;
                let bias = self.b[oc];
                for ot in 0..t_out {
                    let t0 = (ot * self.stride_t) as isize;
                    for of in 0..f_out {
                        let f0 = (of * self.stride_f) as isize;
                        let mut acc = bias;
                        for ic in 0..in_groups {
                            let ic_abs = ic0 + ic;
                            for kt in 0..self.kh {
                                let ti = t0 + kt as isize - pt_l;
                                if ti < 0 || ti >= t_in as isize {
                                    continue;
                                }
                                for kf in 0..self.kw {
                                    let fi = f0 + kf as isize - pf_l;
                                    if fi < 0 || fi >= f_in as isize {
                                        continue;
                                    }
                                    let wi = (ic * self.kh + kt) * self.kw + kf;
                                    let xi = (ti as usize * f_in + fi as usize)
                                        * c_in
                                        + ic_abs;
                                    acc += x[xi] * w[wbase + wi];
                                }
                            }
                        }
                        out[(ot * f_out + of) * c_out + oc] = acc;
                    }
                }
            }
        }
    }
}

/// 1D depthwise conv over time (conformer conv module), kernel
/// `[d_model, kh]` row-major, groups = d_model. Zero left/right pads.
pub struct Conv1dDw {
    pub w: Vec<f32>,
    pub dim: usize,
    pub kh: usize,
    pub pad_left: usize,
    pub pad_right: usize,
}

impl std::fmt::Debug for Conv1dDw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conv1dDw {{ {0} channels, k={1} }}", self.dim, self.kh)
    }
}

impl Conv1dDw {
    /// `x` is `[t, dim]` time-major; result `[t + pad_left + pad_right - kh +
    /// 1, dim]`.
    pub fn forward(&self, x: &[f32], t: usize, out: &mut Vec<f32>) {
        let t_out = t + self.pad_left + self.pad_right - self.kh + 1;
        out.resize(t_out * self.dim, 0.0);
        for ot in 0..t_out {
            let t0 = ot as isize - self.pad_left as isize;
            for c in 0..self.dim {
                let mut acc = 0.0f32;
                for k in 0..self.kh {
                    let ti = t0 + k as isize;
                    if ti < 0 || ti as usize >= t {
                        continue;
                    }
                    acc +=
                        x[ti as usize * self.dim + c] * self.w[c * self.kh + k];
                }
                out[ot * self.dim + c] = acc;
            }
        }
    }
}

pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub dim: usize,
    pub eps: f32,
}

impl std::fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LayerNorm({})", self.dim)
    }
}

impl LayerNorm {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let mut mean = 0.0f32;
        for &v in x {
            mean += v;
        }
        mean /= self.dim as f32;
        let mut var = 0.0f32;
        for &v in x {
            var += (v - mean) * (v - mean);
        }
        var /= self.dim as f32;
        let inv = 1.0 / (var + self.eps).sqrt();
        for i in 0..self.dim {
            out[i] = (x[i] - mean) * inv * self.weight[i] + self.bias[i];
        }
    }
}

/// BatchNorm1d with running stats (NeMo conv block; unused when the
/// model uses conv_norm=layer_norm, kept for completeness).
pub struct BatchNorm1d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub eps: f32,
    pub dim: usize,
}

impl std::fmt::Debug for BatchNorm1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchNorm1d({})", self.dim)
    }
}

impl BatchNorm1d {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        for i in 0..self.dim {
            out[i] = (x[i] - self.mean[i])
                * (self.weight[i] / (self.var[i] + self.eps).sqrt())
                + self.bias[i];
        }
    }
}

/// Load a [out, in] (or transposed [in, out]) matrix as Q8 (dtype
/// 7 or 8) or f32. `name` is the tensor base (e.g. "joint.enc");
/// ".weight"/".bias" are appended. Either `inp` or `out` (or both)
/// may be 0 to infer it from the stored shape: for a transposed-
/// stored [in, out] tensor the gguf dims are `[out_len, rows]`; for
/// [out, in] they are `[in_len, rows]`.
pub fn load_lin(
    gguf: &Gguf,
    name: &str,
    inp: usize,
    out: usize,
) -> Result<Lin> {
    let meta = gguf
        .tensor(&format!("{name}.weight"))
        .with_context(|| format!("GGUF tensor {name}.weight not found"))?;
    if meta.dims.len() != 2 {
        bail!("{name}.weight: expected 2 dims, got {:?}", meta.dims);
    }
    let (a, b) = (meta.dims[0] as usize, meta.dims[1] as usize);
    // gguf dims fastest-first: torch [rows, cols] -> dims [cols, rows].
    let (rows, row_len) = match (inp, out) {
        (0, 0) => (b, a), // fully inferred, transposed storage
        (inp, 0) if a == inp => (b, a), // transposed [in, out] torch
        (0, out) if b == out => (a, b), // direct [out, in] torch
        (inp, out) if a == inp && b == out => (b, a), // transposed
        (inp, out) if b == inp && a == out => (a, b), // direct
        _ => bail!(
            "{name}.weight: shape [{a}, {b}] incompatible with [{inp} -> {out}]"
        ),
    };
    let bias = match gguf.tensor(&format!("{name}.bias")) {
        None => None,
        Some(m) => {
            let v = gguf.read_f32(m)?;
            if v.len() != rows {
                bail!("{name}.bias: {} != {rows}", v.len());
            }
            Some(v)
        }
    };
    let q = if meta.dtype == 7 || meta.dtype == 8 {
        let data = gguf.tensor_data(meta)?;
        let variant = if meta.dtype == 8 {
            Q8Variant::Q8F16
        } else {
            Q8Variant::Q8_0
        };
        Some(Q8Mat::new(data.to_vec(), rows, row_len, variant)?)
    } else {
        None
    };
    let f = if q.is_some() {
        None
    } else {
        Some(gguf.read_f32(meta)?)
    };
    Ok(Lin {
        q,
        f,
        bias,
        out: rows,
        inp: row_len,
    })
}

/// Load a 2D conv weight, torch `[out, in, kh, kw]` (gguf dims
/// fastest-first). `name` is the tensor base (e.g.
/// "encoder.pre_encode.conv.0"); ".weight"/".bias" are appended.
/// `groups` for depthwise convs (groups == in == out).
pub fn load_conv2d(
    gguf: &Gguf,
    name: &str,
    inp: usize,
    out: usize,
    kh: usize,
    kw: usize,
    pad_t: (usize, usize),
    pad_f: (usize, usize),
    stride: usize,
    groups: usize,
) -> Result<Conv2d> {
    let meta = gguf
        .tensor(&format!("{name}.weight"))
        .with_context(|| format!("GGUF tensor {name}.weight not found"))?;
    if meta.dims.len() != 4 {
        bail!("{name}.weight: expected 4 dims, got {:?}", meta.dims);
    }
    let d = meta.dims.iter().map(|&x| x as usize).collect::<Vec<_>>();
    // gguf dims fastest-first: [kw, kh, in/groups, out] == torch reversed.
    let (o, i) = (d[3], d[2]);
    if o != out || i * groups != inp {
        bail!(
            "{name}.weight: shape {d:?} incompatible with [{inp} -> {out}] g{groups}"
        );
    }
    if d[1] != kh || d[0] != kw {
        bail!("{name}.weight: kernel {:?} != {kh}x{kw}", &d[..2]);
    }
    let w = gguf.read_f32(meta)?;
    let bias = match gguf.tensor(&format!("{name}.bias")) {
        None => vec![0.0; o],
        Some(m) => {
            let v = gguf.read_f32(m)?;
            if v.len() != o {
                bail!("{name}.bias: {} != {o}", v.len());
            }
            v
        }
    };
    Ok(Conv2d {
        w,
        b: bias,
        out: o,
        inp,
        kh,
        kw,
        pad_t,
        pad_f,
        stride_t: stride,
        stride_f: stride,
        groups,
    })
}

/// Load a 1D depthwise conv over time, torch `[out, 1, kh]` (gguf dims
/// fastest-first: [kh, 1, out]). `name` is the tensor base; ".weight"
/// is appended.
pub fn load_conv1d_dw(
    gguf: &Gguf,
    name: &str,
    dim: usize,
    kh: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<Conv1dDw> {
    let meta = gguf
        .tensor(&format!("{name}.weight"))
        .with_context(|| format!("GGUF tensor {name}.weight not found"))?;
    if meta.dims.len() != 3 {
        bail!("{name}.weight: expected 3 dims, got {:?}", meta.dims);
    }
    let d = meta.dims.iter().map(|&x| x as usize).collect::<Vec<_>>();
    if d[2] != dim || d[1] != 1 || d[0] != kh {
        bail!("{name}.weight: shape {d:?} incompatible with [{dim}, 1, {kh}]");
    }
    let w = gguf.read_f32(meta)?;
    Ok(Conv1dDw {
        w,
        dim,
        kh,
        pad_left,
        pad_right,
    })
}

pub fn load_linear1d(gguf: &Gguf, name: &str, dim: usize) -> Result<Vec<f32>> {
    let meta = gguf
        .tensor(name)
        .with_context(|| format!("GGUF tensor {name} not found"))?;
    if meta.dims.len() != 1 || meta.dims[0] as usize != dim {
        bail!("{name}: expected 1d of {dim}, got {:?}", meta.dims);
    }
    gguf.read_f32(meta)
}

pub fn load_ln(
    gguf: &Gguf,
    name: &str,
    dim: usize,
    eps: f32,
) -> Result<LayerNorm> {
    let weight = load_linear1d(gguf, &format!("{name}.weight"), dim)?;
    let bias = load_linear1d(gguf, &format!("{name}.bias"), dim)?;
    Ok(LayerNorm {
        weight,
        bias,
        dim,
        eps,
    })
}

/// BatchNorm1d from NeMo state dict (running stats optional).
pub fn load_batchnorm(
    gguf: &Gguf,
    name: &str,
    dim: usize,
    eps: f32,
) -> Result<BatchNorm1d> {
    let weight = load_linear1d(gguf, &format!("{name}.weight"), dim)?;
    let bias = load_linear1d(gguf, &format!("{name}.bias"), dim)?;
    let mean = match gguf.tensor(&format!("{name}.running_mean")) {
        Some(m) => gguf.read_f32(m)?,
        None => vec![0.0; dim],
    };
    let var = match gguf.tensor(&format!("{name}.running_var")) {
        Some(m) => gguf.read_f32(m)?,
        None => vec![1.0; dim],
    };
    Ok(BatchNorm1d {
        weight,
        bias,
        mean,
        var,
        eps,
        dim,
    })
}
