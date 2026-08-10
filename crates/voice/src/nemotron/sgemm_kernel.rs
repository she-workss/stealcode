//! Small, dedicated AVX2/FMA sgemm tuned for the encoder shapes
//! (m up to 4352, k 768..4352, n small 1..128). matrixmultiply's
//! general kernel leaves ~3x on the table for these thin matrices;
//! this one registers 4x8 outputs per FMA pass with an 8-wide k
//! unroll, splits the work over rayon by m-rows, and keeps the
//! n-major input transposed once so B columns are contiguous.

use std::arch::x86_64::*;

use rayon::prelude::*;

use super::gguf::f16_to_f32;

/// `c[m, n] = a[m, k] @ b[k, n]` (all row-major), writing into an
/// existing `c` of exactly `m * n` elements.
/// Uses the AVX2 kernels on capable x86-64, else matrixmultiply.
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
    if m % 8 == 0
        && k % 8 == 0
        && n % 8 == 0
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        // b is used directly: it is `[k, n]` row-major, so a whole
        // 8-column tile per k row is one contiguous 256-bit load.
        // c splits into m/8 disjoint 8-row tiles.
        c.par_chunks_mut(8 * n).enumerate().for_each(|(mi, cpart)| {
            let arows = &a[mi * 8 * k..(mi + 1) * 8 * k];
            // SAFETY: AVX2/FMA detected above; cpart is a disjoint
            // 8-row tile.
            unsafe {
                sgemm_kernel8_avx2(k, n, arows, b, cpart);
            }
        });
    } else if m % 8 == 0
        && k % 8 == 0
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        let mut bt = vec![0.0f32; n * k];
        for kk in 0..k {
            let src = &b[kk * n..(kk + 1) * n];
            for nn in 0..n {
                bt[nn * k + kk] = src[nn];
            }
        }
        let n_threads = std::thread::available_parallelism()
            .map(|x| x.get().min(16))
            .unwrap_or(1)
            .max(1);
        let chunk_rows = m.div_ceil(n_threads).div_ceil(8) * 8;
        let rows_per_thread = chunk_rows;
        c.par_chunks_mut(rows_per_thread * n).enumerate().for_each(
            |(i, c_part)| {
                let m0 = i * rows_per_thread;
                let m1 = (m0 + c_part.len() / n).min(m);
                // SAFETY: c_part is a disjoint row slice; AVX2/FMA checked
                // above.
                unsafe {
                    sgemm_kernel_avx2(
                        m1 - m0,
                        k,
                        n,
                        &a[m0 * k..m1 * k],
                        &bt,
                        c_part,
                    );
                }
            },
        );
    } else {
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

/// Core 8x8: `c[8, n] = a[8, k] @ b[k, n]`, b row-major `[k, n]`.
/// Register block: 8 rows x 8 cols, k unrolled by 8. Per k-unroll step
/// it loads 8 `a` row vectors (8 contiguous k) and 8 `b` tile vectors
/// (one k row each), then fans each a-element out over the 8 b tiles
/// with `vpermps` (mask j selects element j of each a vector). That is
/// 16 loads + 64 FMA per 8 k-steps (0.25 loads/FMA), well past the 1x8
/// kernel's 1.125. Requires m, k, n % 8 == 0.
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_kernel8_avx2(
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    debug_assert!(k % 8 == 0 && n % 8 == 0);
    let k8 = k / 8;
    let n8 = n / 8;
    let masks = [
        _mm256_set1_epi32(0),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(2),
        _mm256_set1_epi32(3),
        _mm256_set1_epi32(4),
        _mm256_set1_epi32(5),
        _mm256_set1_epi32(6),
        _mm256_set1_epi32(7),
    ];
    for nj8 in 0..n8 {
        let mut acc = [_mm256_setzero_ps(); 8];
        for kk8 in 0..k8 {
            let kk = kk8 * 8;
            let mut a8 = [_mm256_setzero_ps(); 8];
            for i in 0..8 {
                a8[i] = _mm256_loadu_ps(a.get_unchecked(i * k + kk));
            }
            let mut b8 = [_mm256_setzero_ps(); 8];
            for j in 0..8 {
                b8[j] =
                    _mm256_loadu_ps(b.get_unchecked((kk + j) * n + nj8 * 8));
            }
            for i in 0..8 {
                let av = a8[i];
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[0]),
                    b8[0],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[1]),
                    b8[1],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[2]),
                    b8[2],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[3]),
                    b8[3],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[4]),
                    b8[4],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[5]),
                    b8[5],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[6]),
                    b8[6],
                    acc[i],
                );
                acc[i] = _mm256_fmadd_ps(
                    _mm256_permutevar8x32_ps(av, masks[7]),
                    b8[7],
                    acc[i],
                );
            }
        }
        for i in 0..8 {
            _mm256_storeu_ps(c.get_unchecked_mut(i * n + nj8 * 8), acc[i]);
        }
    }
}

/// Core: `c[m, n] = a[m, k] * bt[n, k]` (bt row-major `[n, k]`).
/// Register block: 1 row x 8 cols, k unrolled by 8. Each ymm
/// accumulator holds 8 partial sums over k ≡ e (mod 8) for one output
/// column; a horizontal reduction at the end folds them into the final
/// scalar. Requires m, k % 8 == 0; `n` may be any value (the tail
/// columns n % 8 are handled scalar).
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_kernel_avx2(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
) {
    debug_assert!(m % 8 == 0 && k % 8 == 0);
    let k8 = k / 8;
    let n8 = n / 8;
    for mi in 0..m {
        let arow = &a[mi * k..(mi + 1) * k];
        for nj8 in 0..n8 {
            let nj = nj8 * 8;
            let mut acc = [_mm256_setzero_ps(); 8];
            for kk8 in 0..k8 {
                let kk = kk8 * 8;
                let av = _mm256_loadu_ps(arow.get_unchecked(kk));
                for j in 0..8 {
                    let bv =
                        _mm256_loadu_ps(bt.get_unchecked((nj + j) * k + kk));
                    acc[j] = _mm256_fmadd_ps(av, bv, acc[j]);
                }
            }
            for j in 0..8 {
                let s = _mm256_hadd_ps(acc[j], acc[j]);
                let s = _mm256_hadd_ps(s, s);
                let lo = _mm256_castps256_ps128(s);
                let hi = _mm256_extractf128_ps(s, 1);
                let res = _mm_add_ss(lo, hi);
                *c.get_unchecked_mut(mi * n + nj + j) = _mm_cvtss_f32(res);
            }
        }
        // tail columns n % 8 (usually 0; 1..7 for flush tails) — scalar.
        let nj0 = n8 * 8;
        for (j, cj) in c
            .get_unchecked_mut(mi * n + nj0..(mi + 1) * n)
            .iter_mut()
            .enumerate()
        {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += arow[kk] * bt[(nj0 + j) * k + kk];
            }
            *cj = acc;
        }
    }
}

// ---------------------------------------------------------------------------
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
fn read_q8_scale(w: &[u8], base: usize, block_bytes: usize) -> f32 {
    if block_bytes == 34 {
        f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]))
    } else {
        debug_assert_eq!(block_bytes, 36);
        f32::from_le_bytes([w[base], w[base + 1], w[base + 2], w[base + 3]])
    }
}

/// Quantize one activation column `x[:, nj]` (x row-major `[k, n]`) into
/// `xqrow` (`[k]` i8, per-32-block) + `dxrow` (`[nblocks]` scales).
fn quantize_col(
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
    let nblocks = k.div_ceil(32);
    let mut xq = vec![0i8; n * k];
    let mut dx = vec![0.0f32; n * nblocks];
    if n * nblocks >= 1024 {
        xq.par_chunks_mut(k)
            .zip(dx.par_chunks_mut(nblocks))
            .enumerate()
            .for_each(|(nj, (xqrow, dxrow))| {
                quantize_col(x, nj, k, n, nblocks, xqrow, dxrow);
            });
    } else {
        for nj in 0..n {
            let (xqrow, dxrow) = (
                &mut xq[nj * k..(nj + 1) * k],
                &mut dx[nj * nblocks..(nj + 1) * nblocks],
            );
            quantize_col(x, nj, k, n, nblocks, xqrow, dxrow);
        }
    }

    let use_avx2 = m % 8 == 0
        && k % 32 == 0
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma");
    if use_avx2 {
        let tile_rows = 8usize;
        y.par_chunks_mut(tile_rows * n)
            .enumerate()
            .for_each(|(ti, ypart)| {
                let m0 = ti * tile_rows;
                let mut acc = vec![0.0f32; tile_rows * n * 8];
                // SAFETY: AVX2/FMA checked above; ypart is a disjoint
                // 8-row tile.
                unsafe {
                    q8_gemm_tile_avx2(
                        m0,
                        k,
                        n,
                        w,
                        padded_row,
                        block_bytes,
                        qoff,
                        &xq,
                        &dx,
                        nblocks,
                        ypart,
                        &mut acc,
                    );
                }
            });
    } else {
        q8_gemm_scalar(m, k, n, w, padded_row, block_bytes, qoff, x, y);
    }
}

/// Scalar fallback for `q8_gemm` (no AVX2/FMA, or forced for debug).
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

/// AVX2 8-row tile of `q8_gemm`: for each of 8 weight rows and each
/// 32-element k block, load the weight block once and fan it out over all
/// `n` activation columns using the maddubs/madd int8 dot trick.
/// Requires `m % 8 == 0`, `k % 32 == 0`.
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_gemm_tile_avx2(
    m0: usize,
    k: usize,
    n: usize,
    w: &[u8],
    padded_row: usize,
    block_bytes: usize,
    qoff: usize,
    xq: &[i8],
    dx: &[f32],
    nblocks: usize,
    y: &mut [f32],
    acc: &mut [f32],
) {
    debug_assert_eq!(y.len(), 8 * n);
    debug_assert_eq!(acc.len(), 8 * n * 8);
    let ones = _mm256_set1_epi16(1);
    let tile_rows = 8usize;
    let tile_stride = n * 8;
    acc.fill(0.0);
    for b in 0..nblocks {
        let kbase = b * 32;
        let xkbase = xq.as_ptr().add(kbase);
        for i in 0..tile_rows {
            let wbase = (m0 + i) * padded_row + b * block_bytes;
            let wq = _mm256_loadu_si256(
                w.as_ptr().add(wbase + qoff) as *const __m256i
            );
            let ax = _mm256_sign_epi8(wq, wq); // |w|
            let dwv = _mm256_set1_ps(read_q8_scale(w, wbase, block_bytes));
            let arow = acc.as_mut_ptr().add(i * tile_stride);
            for nj in 0..n {
                let xq32 =
                    _mm256_loadu_si256(xkbase.add(nj * k) as *const __m256i);
                let sx = _mm256_sign_epi8(xq32, wq); // x * sign(w)
                let p16 = _mm256_maddubs_epi16(ax, sx);
                let p32 = _mm256_madd_epi16(p16, ones);
                let a = _mm256_loadu_ps(arow.add(nj * 8));
                let s =
                    _mm256_mul_ps(dwv, _mm256_set1_ps(dx[nj * nblocks + b]));
                _mm256_storeu_ps(
                    arow.add(nj * 8),
                    _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(p32), a),
                );
            }
        }
    }
    for i in 0..tile_rows {
        for nj in 0..n {
            let a = _mm256_loadu_ps(acc.as_ptr().add(i * tile_stride + nj * 8));
            let s = _mm256_hadd_ps(a, a);
            let s = _mm256_hadd_ps(s, s);
            let lo = _mm256_castps256_ps128(s);
            let hi = _mm256_extractf128_ps(s, 1);
            let res = _mm_add_ss(lo, hi);
            *y.get_unchecked_mut(i * n + nj) = _mm_cvtss_f32(res);
        }
    }
}
