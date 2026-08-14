//! Small, dedicated AVX2/FMA sgemm tuned for the encoder shapes
//! (m up to 4352, k 768..4352, n small 1..128). matrixmultiply's
//! general kernel leaves ~3x on the table for these thin matrices;
//! this one registers 4x8 outputs per FMA pass with an 8-wide k
//! unroll, splits the work over rayon by m-rows, and keeps the
//! n-major input transposed once so B columns are contiguous.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

use crate::gguf::f16_to_f32;

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
    #[cfg(target_arch = "x86_64")]
    {
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
            return;
        }
        if m % 8 == 0
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
                    // SAFETY: c_part is a disjoint row slice; AVX2/FMA
                    // checked above.
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
            return;
        }
    }
    // SAFETY: a, b, c are valid for m*k, k*n, m*n elements (debug-asserted
    // above; c is written into), with matching row-major strides.
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

/// Core 8x8: `c[8, n] = a[8, k] @ b[k, n]`, b row-major `[k, n]`.
/// Register block: 8 rows x 8 cols, k unrolled by 8. Per k-unroll step
/// it loads 8 `a` row vectors (8 contiguous k) and 8 `b` tile vectors
/// (one k row each), then fans each a-element out over the 8 b tiles
/// with `vpermps` (mask j selects element j of each a vector). That is
/// 16 loads + 64 FMA per 8 k-steps (0.25 loads/FMA), well past the 1x8
/// kernel's 1.125. Requires m, k, n % 8 == 0.
#[cfg(target_arch = "x86_64")]
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
                // SAFETY: i < 8, kk < k, k % 8 == 0: reads 8 floats at
                // i*k + kk, within a's m*k elements.
                a8[i] = unsafe { _mm256_loadu_ps(a.get_unchecked(i * k + kk)) };
            }
            let mut b8 = [_mm256_setzero_ps(); 8];
            for j in 0..8 {
                // SAFETY: (kk + j) < k, nj8 < n8: reads 8 floats at
                // (kk + j) * n + nj8 * 8, within b's k*n elements.
                b8[j] = unsafe {
                    _mm256_loadu_ps(b.get_unchecked((kk + j) * n + nj8 * 8))
                };
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
            // SAFETY: i < 8, nj8 < n8: stores 8 floats at i * n + nj8 * 8,
            // within c's [8, n] tile (c.len() == 8 * n).
            unsafe {
                _mm256_storeu_ps(c.get_unchecked_mut(i * n + nj8 * 8), acc[i])
            };
        }
    }
}

/// Core: `c[m, n] = a[m, k] * bt[n, k]` (bt row-major `[n, k]`).
/// Register block: 1 row x 8 cols, k unrolled by 8. Each ymm
/// accumulator holds 8 partial sums over k ≡ e (mod 8) for one output
/// column; a horizontal reduction at the end folds them into the final
/// scalar. Requires m, k % 8 == 0; `n` may be any value (the tail
/// columns n % 8 are handled scalar).
#[cfg(target_arch = "x86_64")]
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
                // SAFETY: kk < k, k % 8 == 0: reads 8 floats at kk, within
                // the row's k elements.
                let av = unsafe { _mm256_loadu_ps(arow.get_unchecked(kk)) };
                for j in 0..8 {
                    // SAFETY: (nj + j) < n, kk + 8 <= k: reads 8 floats at
                    // (nj + j) * k + kk, within bt's n*k elements.
                    let bv = unsafe {
                        _mm256_loadu_ps(bt.get_unchecked((nj + j) * k + kk))
                    };
                    acc[j] = _mm256_fmadd_ps(av, bv, acc[j]);
                }
            }
            for j in 0..8 {
                let s = _mm256_hadd_ps(acc[j], acc[j]);
                let s = _mm256_hadd_ps(s, s);
                let lo = _mm256_castps256_ps128(s);
                let hi = _mm256_extractf128_ps(s, 1);
                let res = _mm_add_ss(lo, hi);
                // SAFETY: mi < m, nj + j < n: stores within c's [m, n] row
                // (c.len() == m * n).
                unsafe {
                    *c.get_unchecked_mut(mi * n + nj + j) = _mm_cvtss_f32(res)
                };
            }
        }
        // tail columns n % 8 (usually 0; 1..7 for flush tails) - scalar.
        let nj0 = n8 * 8;
        // SAFETY: [mi * n, (mi + 1) * n) is one row of c, within bounds.
        for (j, cj) in
            unsafe { c.get_unchecked_mut(mi * n + nj0..(mi + 1) * n) }
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

    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    {
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
            // Thin batches: a row-sequential kernel with contiguous
            // weight reads (the 8-row tile strides 544 B between rows,
            // which some machines' prefetchers cannot keep up with for
            // n = 1); keep the parallel tile path for wide batches.
            if n <= 4 {
                unsafe {
                    q8_gemm_thin_avx2(
                        m,
                        k,
                        n,
                        w,
                        padded_row,
                        block_bytes,
                        qoff,
                        &xq,
                        &dx,
                        nblocks,
                        y,
                    );
                }
            } else {
                y.par_chunks_mut(tile_rows * n).enumerate().for_each(
                    |(ti, ypart)| {
                        let m0 = ti * tile_rows;
                        let mut acc = vec![0.0f32; tile_rows * n * 8];
                        // SAFETY: AVX2/FMA checked above; ypart is a
                        // disjoint 8-row tile.
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
                    },
                );
            }
            return;
        }
    }
    q8_gemm_scalar(m, k, n, w, padded_row, block_bytes, qoff, x, y);
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
    if m * nblocks <= 4096 {
        // Scalar fallback with a tiny workload: sequential, the rayon
        // split overhead would dominate (see the AVX2 branch).
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

/// AVX2 8-row tile of `q8_gemm`: for each of 8 weight rows and each
/// 32-element k block, load the weight block once and fan it out over all
/// `n` activation columns using the maddubs/madd int8 dot trick.
/// Requires `m % 8 == 0`, `k % 32 == 0`.
#[cfg(target_arch = "x86_64")]
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
        // SAFETY: kbase <= k; xq has n * k elements (k % 32 == 0, so the
        // 32-byte reads via xkbase below stay in bounds for any nj < n).
        let xkbase = unsafe { xq.as_ptr().add(kbase) };
        for i in 0..tile_rows {
            let wbase = (m0 + i) * padded_row + b * block_bytes;
            // SAFETY: wbase + qoff + 32 <= w.len(): each row is
            // padded_row bytes and b < nblocks within it.
            let wq = unsafe {
                _mm256_loadu_si256(
                    w.as_ptr().add(wbase + qoff) as *const __m256i
                )
            };
            let ax = _mm256_sign_epi8(wq, wq); // |w|
            let dwv = _mm256_set1_ps(read_q8_scale(w, wbase, block_bytes));
            // SAFETY: acc has 8 * n * 8 elements (debug_asserted).
            let arow = unsafe { acc.as_mut_ptr().add(i * tile_stride) };
            for nj in 0..n {
                // SAFETY: nj < n, k % 32 == 0: reads 32 bytes at
                // nj * k within xq's n * k elements.
                let xq32 = unsafe {
                    _mm256_loadu_si256(xkbase.add(nj * k) as *const __m256i)
                };
                let sx = _mm256_sign_epi8(xq32, wq); // x * sign(w)
                let p16 = _mm256_maddubs_epi16(ax, sx);
                let p32 = _mm256_madd_epi16(p16, ones);
                // SAFETY: nj < n: reads/writes 8 floats at arow + nj * 8,
                // within acc's tile stride.
                let a = unsafe { _mm256_loadu_ps(arow.add(nj * 8)) };
                let s =
                    _mm256_mul_ps(dwv, _mm256_set1_ps(dx[nj * nblocks + b]));
                unsafe {
                    _mm256_storeu_ps(
                        arow.add(nj * 8),
                        _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(p32), a),
                    )
                };
            }
        }
    }
    for i in 0..tile_rows {
        for nj in 0..n {
            // SAFETY: i < 8, nj < n: reads within acc's bounds.
            let a = unsafe {
                _mm256_loadu_ps(acc.as_ptr().add(i * tile_stride + nj * 8))
            };
            let s = _mm256_hadd_ps(a, a);
            let s = _mm256_hadd_ps(s, s);
            let lo = _mm256_castps256_ps128(s);
            let hi = _mm256_extractf128_ps(s, 1);
            let res = _mm_add_ss(lo, hi);
            // SAFETY: i < 8 (tile_rows), nj < n: y has 8 * n elements.
            *unsafe { y.get_unchecked_mut(i * n + nj) } = _mm_cvtss_f32(res);
        }
    }
}

/// Rows per rayon chunk for the thin kernel: each chunk streams its
/// weight rows contiguously so the prefetcher and the parallel lanes both
/// help; the chunk is shrunk for small `m` so all cores stay busy
/// (m = 512 would otherwise split into 2 chunks and leave 6 threads
/// idle), but kept large enough that per-task setup stays negligible.
fn thin_chunk_rows(m: usize) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .max(1);
    let per_thread = m.div_ceil(threads).max(32);
    256usize.min(per_thread.div_ceil(8) * 8).max(32)
}

/// AVX2 row-sequential kernel for thin batches (`n <= 4`). Each output
/// row's weights are read contiguously (unlike the 8-row tile, which
/// strides `padded_row` bytes between rows - a pattern that latency-bound
/// machines cannot hide for n = 1). For each row, every 32-element k
/// block is loaded once and fanned out over the `n` columns via the
/// maddubs/madd int8 dot trick, accumulating in n separate 8-lane
/// registers (partial sums over k mod 8). The block loop is unrolled by
/// 4 with one accumulator set per unroll step, so the accumulator chain
/// is 4x shorter and 8 block loads stay in flight per row (MLP). Rows
/// are split over rayon in contiguous chunks of [`THIN_CHUNK_ROWS`].
/// Requires `m % 8 == 0`, `k % 32 == 0`, `n <= 4`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_gemm_thin_avx2(
    m: usize,
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
) {
    debug_assert!(m % 8 == 0 && k % 32 == 0 && n <= 4);
    let ones = _mm256_set1_epi16(1);
    let chunk_rows = thin_chunk_rows(m);
    y.par_chunks_mut(chunk_rows * n).enumerate().for_each(
        |(ci, ypart)| {
            let i0 = ci * chunk_rows;
            let rows = ypart.len() / n;
            for i in 0..rows {
                // Software-prefetch the next row's weights into L1: the
                // row stream (34 B/block stride) is too irregular for the
                // hardware prefetcher to track at DRAM latencies, and the
                // per-block loads otherwise stall the pipeline (the
                // in-context GEMMs read cold weights every call).
                if i + 1 < rows {
                    // SAFETY: i + 1 < rows, so (i0 + i + 1) * padded_row is
                    // within w's m * padded_row bytes.
                    unsafe {
                        _mm_prefetch(
                            w.as_ptr().add((i0 + i + 1) * padded_row)
                                as *const i8,
                            _MM_HINT_T0,
                        )
                    };
                }
                let rowbase = (i0 + i) * padded_row;
                let mut acc = [[_mm256_setzero_ps(); 4]; 4];
                let mut b = 0;
                while b + 4 <= nblocks {
                    // Issue the 4 weight block loads up front (independent,
                    // so they pipeline), then the per-column dot products
                    // into separate accumulator sets.
                    let mut wq = [_mm256_setzero_si256(); 4];
                    let mut dw = [_mm256_setzero_ps(); 4];
                    for u in 0..4 {
                        let wbase = rowbase + (b + u) * block_bytes;
                        // SAFETY: wbase + qoff + 32 <= w.len(): each row is
                        // padded_row bytes and b + u < nblocks within it.
                        wq[u] = unsafe {
                            _mm256_loadu_si256(
                                w.as_ptr().add(wbase + qoff) as *const __m256i
                            )
                        };
                        dw[u] = _mm256_set1_ps(read_q8_scale(
                            w, wbase, block_bytes,
                        ));
                    }
                    let ax0 = _mm256_sign_epi8(wq[0], wq[0]); // |w|
                    let ax1 = _mm256_sign_epi8(wq[1], wq[1]);
                    let ax2 = _mm256_sign_epi8(wq[2], wq[2]);
                    let ax3 = _mm256_sign_epi8(wq[3], wq[3]);
                    for nj in 0..n {
                        // SAFETY: nj < n, b + u < nblocks, k % 32 == 0:
                        // reads 32 bytes at nj * k + (b + u) * 32 within
                        // xq's n * k elements.
                        let xq0 = unsafe {
                            _mm256_loadu_si256(
                                xq.as_ptr().add(nj * k + b * 32)
                                    as *const __m256i
                            )
                        };
                        let xq1 = unsafe {
                            _mm256_loadu_si256(
                                xq.as_ptr().add(nj * k + (b + 1) * 32)
                                    as *const __m256i
                            )
                        };
                        let xq2 = unsafe {
                            _mm256_loadu_si256(
                                xq.as_ptr().add(nj * k + (b + 2) * 32)
                                    as *const __m256i
                            )
                        };
                        let xq3 = unsafe {
                            _mm256_loadu_si256(
                                xq.as_ptr().add(nj * k + (b + 3) * 32)
                                    as *const __m256i
                            )
                        };
                        let dxb = nj * nblocks + b;
                        let s0 = _mm256_mul_ps(
                            dw[0],
                            _mm256_set1_ps(dx[dxb]),
                        );
                        let s1 = _mm256_mul_ps(
                            dw[1],
                            _mm256_set1_ps(dx[dxb + 1]),
                        );
                        let s2 = _mm256_mul_ps(
                            dw[2],
                            _mm256_set1_ps(dx[dxb + 2]),
                        );
                        let s3 = _mm256_mul_ps(
                            dw[3],
                            _mm256_set1_ps(dx[dxb + 3]),
                        );
                        let sx0 = _mm256_sign_epi8(xq0, wq[0]);
                        let p16 = _mm256_maddubs_epi16(ax0, sx0);
                        let p32 = _mm256_madd_epi16(p16, ones);
                        acc[0][nj] = _mm256_fmadd_ps(
                            s0,
                            _mm256_cvtepi32_ps(p32),
                            acc[0][nj],
                        );
                        let sx1 = _mm256_sign_epi8(xq1, wq[1]);
                        let p16 = _mm256_maddubs_epi16(ax1, sx1);
                        let p32 = _mm256_madd_epi16(p16, ones);
                        acc[1][nj] = _mm256_fmadd_ps(
                            s1,
                            _mm256_cvtepi32_ps(p32),
                            acc[1][nj],
                        );
                        let sx2 = _mm256_sign_epi8(xq2, wq[2]);
                        let p16 = _mm256_maddubs_epi16(ax2, sx2);
                        let p32 = _mm256_madd_epi16(p16, ones);
                        acc[2][nj] = _mm256_fmadd_ps(
                            s2,
                            _mm256_cvtepi32_ps(p32),
                            acc[2][nj],
                        );
                        let sx3 = _mm256_sign_epi8(xq3, wq[3]);
                        let p16 = _mm256_maddubs_epi16(ax3, sx3);
                        let p32 = _mm256_madd_epi16(p16, ones);
                        acc[3][nj] = _mm256_fmadd_ps(
                            s3,
                            _mm256_cvtepi32_ps(p32),
                            acc[3][nj],
                        );
                    }
                    b += 4;
                }
                while b < nblocks {
                    let wbase = rowbase + b * block_bytes;
                    // SAFETY: wbase + qoff + 32 <= w.len(): each row is
                    // padded_row bytes and b < nblocks within it.
                    let wq = unsafe {
                        _mm256_loadu_si256(
                            w.as_ptr().add(wbase + qoff) as *const __m256i
                        )
                    };
                    let ax = _mm256_sign_epi8(wq, wq); // |w|
                    let dwv =
                        _mm256_set1_ps(read_q8_scale(w, wbase, block_bytes));
                    for nj in 0..n {
                        // SAFETY: nj < n, b < nblocks, k % 32 == 0: reads
                        // 32 bytes at nj * k + b * 32 within xq's n * k
                        // elements.
                        let xq32 = unsafe {
                            _mm256_loadu_si256(
                                xq.as_ptr().add(nj * k + b * 32)
                                    as *const __m256i
                            )
                        };
                        let sx = _mm256_sign_epi8(xq32, wq); // x * sign(w)
                        let p16 = _mm256_maddubs_epi16(ax, sx);
                        let p32 = _mm256_madd_epi16(p16, ones);
                        let s = _mm256_mul_ps(
                            dwv,
                            _mm256_set1_ps(dx[nj * nblocks + b]),
                        );
                        acc[0][nj] = _mm256_fmadd_ps(
                            s,
                            _mm256_cvtepi32_ps(p32),
                            acc[0][nj],
                        );
                    }
                    b += 1;
                }
                for nj in 0..n {
                    let a = _mm256_add_ps(
                        _mm256_add_ps(acc[0][nj], acc[1][nj]),
                        _mm256_add_ps(acc[2][nj], acc[3][nj]),
                    );
                    let s = _mm256_hadd_ps(a, a);
                    let s = _mm256_hadd_ps(s, s);
                    let lo = _mm256_castps256_ps128(s);
                    let hi = _mm256_extractf128_ps(s, 1);
                    let res = _mm_add_ss(lo, hi);
                    // SAFETY: i < rows, nj < n: ypart has rows * n
                    // elements.
                    *unsafe { ypart.get_unchecked_mut(i * n + nj) } =
                        _mm_cvtss_f32(res);
                }
            }
        },
    );
}
