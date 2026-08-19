//! GEMM kernels for the encoder shapes (m up to 4352, k 768..4352,
//! n small 1..128). The quantized int8 path (`q8_gemm`) is the
//! encoder's hot loop and runs entirely on the portable `std::simd`
//! kernels in `simd_kernel` (one implementation, vectorized by LLVM
//! for whatever the target supports — AVX2/AVX-512 on x86-64, NEON/
//! SVE on ARM, ...). The f32 path (`gemm_into`) uses the same
//! portable kernels.

use rayon::prelude::*;

use crate::gguf::f16_to_f32;

/// `c[m, n] = a[m, k] @ b[k, n]` (all row-major), writing into an
/// existing `c` of exactly `m * n` elements.
/// Uses the std::simd kernels.
#[allow(unsafe_code, unused_unsafe)]
pub fn gemm_into(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    crate::simd_kernel::gemm_simd_into(m, k, n, a, b, c);
}

// Quantized int8 GEMM (llama.cpp-style q8_0 x q8_0 vec-dot), no f32
// dequantization of the weights. Weights stay as stored (block layout,
// `padded_row` bytes per row), activations are quantized per 32-element
// block (`scale = max|x| / 127`) and the dot product runs in int8 with a
// per-block f32 rescale. This removes the f32 dequant cache entirely
// (the 3 GB -> ~700 MB regression) and is memory-bound-friendlier for the
// thin encoder batches (weight bytes are read once per GEMM instead of
// being widened to f32).

/// Q8 block scale: block_bytes 34 -> Q8F16 (f16 d + i8 x32, this model's
/// layout), 36 -> Q8_0 (f32 d + i8 x32). `base` points at the block's
/// first byte.
pub(crate) fn read_q8_scale(w: &[u8], base: usize, block_bytes: usize) -> f32 {
    if block_bytes == 34 {
        f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]))
    } else {
        debug_assert_eq!(block_bytes, 36);
        f32::from_le_bytes([w[base], w[base + 1], w[base + 2], w[base + 3]])
    }
}

/// Quantize one activation column `x[:, nj]` (x row-major `[k, n]`) into
/// `xqrow` (`[k]` i8, per-32-block) + `dxrow` (`[nblocks]` scales).
pub(crate) fn quantize_col(
    x: &[f32],
    nj: usize,
    k: usize,
    n: usize,
    nblocks: usize,
    xqrow: &mut [i8],
    dxrow: &mut [f32],
) {
    for b in 0..nblocks {
        let k0 = b * 32;
        let len = 32usize.min(k - k0);
        let mut maxv = 0.0f32;
        for j in 0..len {
            let v = x[(k0 + j) * n + nj].abs();
            if v > maxv {
                maxv = v;
            }
        }
        let s = if maxv == 0.0 { 1.0 } else { maxv / 127.0 };
        dxrow[b] = s;
        if maxv != 0.0 {
            let inv = 1.0 / s;
            for j in 0..len {
                let q = (x[(k0 + j) * n + nj] * inv).round() as i8;
                xqrow[k0 + j] = q;
            }
        } else {
            for j in 0..len {
                xqrow[k0 + j] = 0;
            }
        }
    }
}

/// `c[m, n] = a[m, k] @ b[k, n]` where `a` is a quantized matrix in the
/// Q8 block layout used by `Q8Mat` (`w` = `m` rows of `padded_row` bytes,
/// each `k.div_ceil(32)` blocks of `block_bytes`, scale at `qoff`-byte
/// offset before the 32 i8 values) and `b` = `x` is f32 `[k, n]`
/// row-major. Writes `[m, n]` row-major into `y` (beta = 0).
/// Uses the std::simd kernels.
#[allow(clippy::too_many_arguments)]
pub fn q8_gemm(
    m: usize,
    k: usize,
    n: usize,
    w: &[u8],
    padded_row: usize,
    block_bytes: usize,
    qoff: usize,
    x: &[f32],
    y: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(w.len(), m * padded_row);
    debug_assert_eq!(x.len(), k * n);
    debug_assert_eq!(y.len(), m * n);
    crate::simd_kernel::q8_gemm_simd(
        m,
        k,
        n,
        w,
        padded_row,
        block_bytes,
        qoff,
        x,
        y,
    );
}

/// Reference implementation of `q8_gemm` (kept as the test oracle for
/// the SIMD kernel).
#[allow(clippy::too_many_arguments)]
pub fn q8_gemm_scalar(
    m: usize,
    k: usize,
    n: usize,
    w: &[u8],
    padded_row: usize,
    block_bytes: usize,
    qoff: usize,
    x: &[f32],
    y: &mut [f32],
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    let nblocks = k.div_ceil(32);
    let mut xq = vec![0i8; n * k];
    let mut dx = vec![0.0f32; n * nblocks];
    for nj in 0..n {
        let (xqrow, dxrow) = (
            &mut xq[nj * k..(nj + 1) * k],
            &mut dx[nj * nblocks..(nj + 1) * nblocks],
        );
        quantize_col(x, nj, k, n, nblocks, xqrow, dxrow);
    }
    if m * nblocks <= 4096 {
        // Scalar fallback with a tiny workload: sequential, the rayon
        // split overhead would dominate.
        for i in 0..m {
            let yrow = &mut y[i * n..(i + 1) * n];
            for nj in 0..n {
                let mut acc = 0.0f32;
                for b in 0..nblocks {
                    let wbase = i * padded_row + b * block_bytes;
                    let dw = read_q8_scale(w, wbase, block_bytes);
                    let s = dw * dx[nj * nblocks + b];
                    let k0 = b * 32;
                    let len = 32usize.min(k - k0);
                    let wb = i * padded_row + b * block_bytes + qoff;
                    let xb = nj * k + k0;
                    let mut dot = 0.0f32;
                    for j in 0..len {
                        dot += (w[wb + j] as i8 as f32) * (xq[xb + j] as f32);
                    }
                    acc += s * dot;
                }
                yrow[nj] = acc;
            }
        }
    } else {
        y.par_chunks_mut(n).enumerate().for_each(|(i, yrow)| {
            for nj in 0..n {
                let mut acc = 0.0f32;
                for b in 0..nblocks {
                    let wbase = i * padded_row + b * block_bytes;
                    let dw = read_q8_scale(w, wbase, block_bytes);
                    let s = dw * dx[nj * nblocks + b];
                    let k0 = b * 32;
                    let len = 32usize.min(k - k0);
                    let wb = i * padded_row + b * block_bytes + qoff;
                    let xb = nj * k + k0;
                    let mut dot = 0.0f32;
                    for j in 0..len {
                        dot += (w[wb + j] as i8 as f32) * (xq[xb + j] as f32);
                    }
                    acc += s * dot;
                }
                yrow[nj] = acc;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 -> IEEE-754 half bits (for building valid q8_0 scale bytes).
    fn half_f32_to_u16(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3ff;
        if exp <= 0 {
            return sign;
        }
        if exp >= 31 {
            return sign | 0x7c00;
        }
        sign | ((exp as u16) << 10) | (mant as u16)
    }

    fn run_case(m: usize, k: usize, n: usize) {
        let nblocks = k.div_ceil(32);
        let padded_row = nblocks * 34;
        let block_bytes = 34;
        let mut w = vec![0u8; m * padded_row];
        let mut x = vec![0.0f32; k * n];
        for i in 0..m {
            for b in 0..nblocks {
                let wbase = i * padded_row + b * block_bytes;
                let scale = if (i + b) % 2 == 0 { 0.0125 } else { 0.03125 };
                let f16 = half_f32_to_u16(scale);
                w[wbase] = f16 as u8;
                w[wbase + 1] = (f16 >> 8) as u8;
                for j in 0..32 {
                    w[wbase + 2 + j] = ((i * 31 + b * 7 + j * 13) % 251) as u8;
                }
            }
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 3) % 101) as f32 / 17.0 - 1.0;
        }
        let mut y = vec![0.0f32; m * n];
        let mut y_ref = vec![0.0f32; m * n];
        q8_gemm(m, k, n, &w, padded_row, 34, 2, &x, &mut y);
        q8_gemm_scalar(m, k, n, &w, padded_row, 34, 2, &x, &mut y_ref);
        for i in 0..m * n {
            let d = (y[i] - y_ref[i]).abs();
            let tol = 1e-3 * (1.0 + y_ref[i].abs());
            assert!(
                d <= tol,
                "m={m} k={k} n={n} [{i}]: {} vs {}",
                y[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn vnni_matches_scalar() {
        run_case(8, 64, 1);
        run_case(16, 128, 2);
        run_case(32, 512, 4);
        run_case(64, 512, 8);
        run_case(128, 768, 14);
        run_case(128, 2048, 16);
        run_case(512, 2048, 1);
        run_case(512, 4352, 16);
    }
}
