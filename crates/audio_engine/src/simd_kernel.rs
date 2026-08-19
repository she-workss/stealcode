//! Portable SIMD implementations of the GEMMs used by the streaming
//! encoder (`std::simd`, nightly). One implementation, vectorized per
//! target by LLVM.
//!
//! Vector width: with `S = 16` f32 lanes the vectorized ops are
//! 512-bit on AVX-512 and 2×256-bit on AVX2. This file deliberately
//! ships only two x86 tiers — `avx2+fma` and baseline (SSE2) — and no
//! AVX-512 tier: the target machines guarantee AVX2/SSE but not
//! AVX-512, and AVX2 performance is sufficient (see the dispatch
//! notes below). With the workspace baseline `-Ctarget-cpu=x86-64-v3`
//! (AVX2+FMA) the baseline path already lowers `Simd::<f32, 16>` to
//! 2×256-bit, so on AVX2 machines the two tiers generate identical
//! code; the runtime dispatch still matters on SSE2-only machines and
//! on non-x86 targets (NEON/SVE).
//!
//! Q8 path (`q8_gemm_simd`): there is no VNNI (`vpdpbusd`) equivalent
//! in `std::simd`, so the dot product uses the int8 → int16 → int32
//! widen chain. Each weight block is loaded once per row and fanned
//! out over the `n` columns; each (row, column) accumulator holds 8
//! partial sums over `k mod 8` (an 8-lane f32 vector in a stack
//! buffer), so no horizontal reduce runs per block. The int16 product
//! of two q8 values fits i16 (127² + 127² < 32767), so each block's
//! 32 products are folded pairwise in i16 before widening to f32 —
//! one fewer wide conversion than the plain i32 widen chain. Rows are
//! streamed sequentially in contiguous chunks (one weight stream per
//! rayon task, no per-task allocations) with a software prefetch of
//! the next row, which keeps the hardware prefetcher engaged for the
//! thin streaming batches. `n == 1` rows keep a single register
//! accumulator (no memory round-trip per block).
//!
//! f32 path (`gemm_simd_into`): pure `mul_add` on f32, which `std::simd`
//! expresses natively. On AVX2 the 16-lane row kernel processes 16
//! columns at once as 2×256-bit with 17 loads per 256 FMA (4.25 B/FMA
//! of L1 traffic) — half the L1 bytes per FMA of the manual 8×8 AVX2
//! tile (16 × 32 B per 64 FMA = 8 B/FMA), so the portable f32 kernel
//! competes head-to-head with the hand-written AVX2 path.

#[cfg(target_arch = "x86_64")]
use std::is_x86_feature_detected;
use std::simd::{StdFloat, prelude::*};

use rayon::prelude::*;

use crate::sgemm_kernel::{quantize_col, read_q8_scale};

const L: usize = 32;
const S: usize = 16;

/// Generates a safe dispatcher for a `std::simd` kernel:
///
/// - on x86_64, if `avx2 + fma` are detected at runtime the work goes to
///   `$avx2` (compiled with `#[target_feature(enable = "avx2,fma")]` so LLVM
///   lowers the `std::simd` ops to 256-bit vectors even when the crate baseline
///   is SSE2, i.e. without `-Ctarget-cpu=native`);
/// - otherwise `$impl` runs (baseline codegen: SSE2 on x86, NEON/SVE on ARM) —
///   correct everywhere, just slower.
///
/// There is deliberately no AVX-512 tier (see the module docs): the
/// target machines guarantee AVX2 but not AVX-512, and the AVX2 tier
/// is sufficient. The GEMMs (`q8_gemm_simd`, `gemm_simd_into`) and
/// `softmax_v` do not use this macro — their rayon splits must stay
/// outside the `#[target_feature]` boundary (see their dispatchers).
macro_rules! dispatch_avx2 {
    ($pub:ident, $impl:ident, $avx2:ident, $ret:ty, [$($arg:ident: $ty:ty),*]) => {
        pub fn $pub($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: the avx2+fma feature set was runtime-verified above.
                    return unsafe { $avx2($($arg),*) };
                }
            }
            $impl($($arg),*)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx2($($arg: $ty),*) -> $ret {
            $impl($($arg),*)
        }
    };
}

/// `y[m, n] = W[m, k] @ x[k, n]` in the Q8 block layout (see
/// `sgemm_kernel::q8_gemm` for the format). Requires `m % 8 == 0`,
/// `k % 32 == 0`, `n <= 16`; other shapes fall back to the scalar
/// kernel in `sgemm_kernel`.
#[allow(
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::ptr_as_ptr,
    clippy::unnecessary_cast,
    clippy::suboptimal_flops,
    clippy::needless_range_loop
)]
pub fn q8_gemm_simd(
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
    debug_assert!(m % 8 == 0 && k % L == 0 && n <= 16);
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    if m % 8 != 0 || k % L != 0 || n > 16 {
        // Shapes outside the 8-row/32-block/16-column fast path.
        crate::sgemm_kernel::q8_gemm_scalar(
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
        return;
    }
    let nblocks = k / L;
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
    q8_gemm_simd_q(
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

/// Kernel half of `q8_gemm_simd` over pre-quantized activations
/// (`xq`/`dx` produced by `sgemm_kernel::quantize_col`). Exposed so
/// callers sharing one input across several matrices (attention q/k/v)
/// quantize once and reuse the result.
pub(crate) fn q8_gemm_simd_q(
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
    // n == 1 (LSTM matvec, per-token): single-column rows are cheap
    // enough that the chunked rayon split below stays worthwhile for
    // any realistic m (each chunk streams its rows sequentially).
    // One contiguous weight stream per rayon task: chunks of 32..96
    // rows (short streams drop to DRAM latency-bound speeds, long ones
    // starve the other cores; 32-96 keeps all cores busy at m = 768..
    // 4352 while leaving the hardware prefetcher enough lead).
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .max(1);
    let per_thread = m.div_ceil(threads);
    let chunk_rows = per_thread.clamp(32, 96).div_ceil(8) * 8;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the avx2+fma feature set was runtime-verified
            // above; ypart is a disjoint chunk of y.
            y.par_chunks_mut(chunk_rows * n).enumerate().for_each(
                |(ci, ypart)| unsafe {
                    q8_gemm_simd_chunk_avx2(
                        ci,
                        chunk_rows,
                        k,
                        n,
                        nblocks,
                        padded_row,
                        block_bytes,
                        qoff,
                        w,
                        xq,
                        dx,
                        ypart,
                    );
                },
            );
            return;
        }
    }
    y.par_chunks_mut(chunk_rows * n)
        .enumerate()
        .for_each(|(ci, ypart)| {
            q8_gemm_simd_chunk_impl(
                ci,
                chunk_rows,
                k,
                n,
                nblocks,
                padded_row,
                block_bytes,
                qoff,
                w,
                xq,
                dx,
                ypart,
            );
        });
}

/// Per-chunk rows `[ci * chunk_rows, ci * chunk_rows + rows)` of
/// `q8_gemm_simd`. Kept as a separate fn from the rayon split so the
/// SIMD body can be compiled under `#[target_feature]` (the rayon
/// closure machinery would otherwise be a baseline-codegen boundary).
///
/// Row-sequential with per-block fanout, mirroring the hand-written
/// 8-row tile structure: each weight block is loaded once per row and
/// fanned out over the `n` columns, and each (row, column)
/// accumulator holds 8 partial sums over `k mod 8` (an 8-lane f32
/// vector in a stack buffer), so no horizontal reduce runs per block
/// and the int8 chain stays in registers. The int16 product of two
/// q8 values fits i16 (127² + 127² < 32767), so the block's 32
/// products are folded pairwise in i16 before widening to f32 —
/// one fewer wide conversion than the plain i32 widen chain.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
/// Software-prefetch the next row's first cache lines (and its page
/// walk) while the current row's blocks are still streaming: the
/// 34 B/block stride is too irregular for the hardware prefetcher to
/// track at DRAM/L3 latencies, so the per-block loads otherwise stall
/// the pipeline. One `prefetch_read_data` per row is enough — it opens
/// the row's line stream and walks its page early.
#[inline(always)]
fn prefetch_next_row(
    w: &[u8],
    rowbase: usize,
    padded_row: usize,
    rows_left: usize,
) {
    if rows_left > 1 {
        // SAFETY: rows_left > 1: (rowbase + padded_row) is the next
        // row within w's m * padded_row bytes.
        unsafe {
            core::intrinsics::prefetch_read_data::<i8, 1>(
                w.as_ptr().add(rowbase + padded_row) as *const i8,
            );
        }
    }
}

#[inline(always)]
fn q8_gemm_simd_chunk_impl(
    ci: usize,
    chunk_rows: usize,
    k: usize,
    n: usize,
    nblocks: usize,
    padded_row: usize,
    block_bytes: usize,
    qoff: usize,
    w: &[u8],
    xq: &[i8],
    dx: &[f32],
    ypart: &mut [f32],
) {
    let i0 = ci * chunk_rows;
    let rows = ypart.len() / n;
    if n == 1 {
        // SAFETY: rows == ypart.len(): each chunk writes rows scalars.
        let yrow =
            unsafe { std::slice::from_raw_parts_mut(ypart.as_mut_ptr(), rows) };
        // SAFETY: k % 32 == 0: xq has k elements.
        let xqrow = unsafe { std::slice::from_raw_parts(xq.as_ptr(), k) };
        let dxrow = &dx[..nblocks];
        for i in 0..rows {
            prefetch_next_row(w, (i0 + i) * padded_row, padded_row, rows - i);
            let rowbase = (i0 + i) * padded_row;
            let mut acc = Simd::<f32, 8>::splat(0.0);
            for b in 0..nblocks {
                let wbase = rowbase + b * block_bytes;
                let dw = read_q8_scale(w, wbase, block_bytes);
                // SAFETY: b < nblocks, each row is padded_row bytes:
                // the 32 i8 values at wbase + qoff are within w.
                let wq = unsafe {
                    Simd::<i8, L>::from_slice(std::slice::from_raw_parts(
                        w.as_ptr().add(wbase + qoff) as *const i8,
                        L,
                    ))
                };
                // SAFETY: b < nblocks: 32 i8 values at b * L within
                // xqrow's k elements.
                let xqj = unsafe {
                    Simd::<i8, L>::from_slice(std::slice::from_raw_parts(
                        xqrow.as_ptr().add(b * L),
                        L,
                    ))
                };
                let p16 = wq.cast::<i16>() * xqj.cast::<i16>();
                let p = p16.extract::<0, 16>() + p16.extract::<16, 16>();
                let p32 = p.cast::<i32>().cast::<f32>();
                let s8 = p32.extract::<0, 8>() + p32.extract::<8, 8>();
                acc = s8.mul_add(Simd::<f32, 8>::splat(dw * dxrow[b]), acc);
            }
            yrow[i] = acc.reduce_sum();
        }
        return;
    }
    let mut acc = [0.0f32; 16 * 8];
    for i in 0..rows {
        prefetch_next_row(w, (i0 + i) * padded_row, padded_row, rows - i);
        let rowbase = (i0 + i) * padded_row;
        let arow = &mut acc[..n * 8];
        arow.fill(0.0);
        for b in 0..nblocks {
            let wbase = rowbase + b * block_bytes;
            let dw = read_q8_scale(w, wbase, block_bytes);
            // SAFETY: b < nblocks, each row is padded_row bytes: the
            // 32 i8 values at wbase + qoff are within w.
            let wq = unsafe {
                Simd::<i8, L>::from_slice(std::slice::from_raw_parts(
                    w.as_ptr().add(wbase + qoff) as *const i8,
                    L,
                ))
            };
            let w16 = wq.cast::<i16>();
            for nj in 0..n {
                // SAFETY: nj < n, b < nblocks, k % 32 == 0: the 32 i8
                // values at nj * k + b * L are within xq's n * k
                // elements.
                let xqj = unsafe {
                    Simd::<i8, L>::from_slice(std::slice::from_raw_parts(
                        xq.as_ptr().add(nj * k + b * L) as *const i8,
                        L,
                    ))
                };
                // i16 pairwise fold: lane j gets p[j] + p[j + 16],
                // which fits i16 (see the docs).
                let p16 = w16 * xqj.cast::<i16>();
                let p = p16.extract::<0, 16>() + p16.extract::<16, 16>();
                let p32 = p.cast::<i32>().cast::<f32>();
                // 8 partial sums over k mod 8: lane j gets the sum of
                // lanes j, j + 8 of p32.
                let s8 = p32.extract::<0, 8>() + p32.extract::<8, 8>();
                let s = dw * dx[nj * nblocks + b];
                // SAFETY: nj < n: arow has n * 8 elements.
                let a8 = unsafe {
                    Simd::<f32, 8>::from_slice(std::slice::from_raw_parts(
                        arow.as_ptr().add(nj * 8),
                        8,
                    ))
                };
                let r8 = s8.mul_add(Simd::<f32, 8>::splat(s), a8);
                arow[nj * 8..nj * 8 + 8].copy_from_slice(&r8.to_array());
            }
        }
        for nj in 0..n {
            // SAFETY: nj < n: arow has n * 8 elements.
            let a8 = unsafe {
                Simd::<f32, 8>::from_slice(std::slice::from_raw_parts(
                    arow.as_ptr().add(nj * 8),
                    8,
                ))
            };
            // SAFETY: i < rows, nj < n: ypart has rows * n elements.
            unsafe {
                *ypart.get_unchecked_mut(i * n + nj) = a8.reduce_sum();
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn q8_gemm_simd_chunk_avx2(
    ci: usize,
    chunk_rows: usize,
    k: usize,
    n: usize,
    nblocks: usize,
    padded_row: usize,
    block_bytes: usize,
    qoff: usize,
    w: &[u8],
    xq: &[i8],
    dx: &[f32],
    ypart: &mut [f32],
) {
    q8_gemm_simd_chunk_impl(
        ci,
        chunk_rows,
        k,
        n,
        nblocks,
        padded_row,
        block_bytes,
        qoff,
        w,
        xq,
        dx,
        ypart,
    );
}

/// `c[m, n] = a[m, k] @ b[k, n]` (all row-major), portable twin of
/// `sgemm_kernel::gemm_into`. No shape restrictions: any m, k, n.
///
/// Two kernels, mirroring the arch dispatch:
///  - 8-row × 8-column tiles (`gemm_tile8`): each k-chunk loads 8 `a` row
///    vectors and 8 `b` column vectors (from the `[n, k]` transpose), then fans
///    each `a` element out over the 8 `b` vectors via a lane-broadcast
///    `swizzle_dyn` + `mul_add` — 16 loads per 64 FMAs (0.25 loads/FMA), the
///    same ratio as the hand-written AVX2 kernel; on AVX2 the 16-lane FMAs
///    lower to 2×256-bit (equal width to the arch kernel's 256-bit ones);
///  - a 1×S row kernel for the remainder rows/columns (and any shape that is
///    not 8-row aligned).
///
/// The rayon row-split lives in this dispatcher (not inside the SIMD
/// body): the rayon closure machinery is monomorphized as separate
/// baseline-codegen functions, so the per-chunk SIMD work is a
/// `#[target_feature]` fn (`gemm_simd_chunk_avx2`) called from the
/// closure instead of being inlined into it.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gemm_simd_into(
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
    let mut bt = Vec::new();
    if n % S != 0 || m % 8 != 0 {
        bt.resize(n * k, 0.0f32);
        for kk in 0..k {
            let src = &b[kk * n..(kk + 1) * n];
            for nn in 0..n {
                bt[nn * k + kk] = src[nn];
            }
        }
    }
    let chunk_rows = 256usize.min(m).max(1);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the avx2+fma feature set was runtime-verified
            // above; cpart is a disjoint chunk of c.
            c.par_chunks_mut(chunk_rows * n).enumerate().for_each(
                |(ci, cpart)| unsafe {
                    gemm_simd_chunk_avx2(
                        ci, chunk_rows, k, n, a, b, &bt, cpart,
                    );
                },
            );
            return;
        }
    }
    c.par_chunks_mut(chunk_rows * n)
        .enumerate()
        .for_each(|(ci, cpart)| {
            gemm_simd_chunk_impl(ci, chunk_rows, k, n, a, b, &bt, cpart);
        });
}

/// Per-chunk rows `[ci * chunk_rows, ci * chunk_rows + rows)` of
/// `gemm_simd_into`. Kept as a separate fn from the rayon split so the
/// SIMD body can be compiled under `#[target_feature]` (the rayon
/// closure machinery would otherwise be a baseline-codegen boundary).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline(always)]
fn gemm_simd_chunk_impl(
    ci: usize,
    chunk_rows: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    bt: &[f32],
    cpart: &mut [f32],
) {
    let i0 = ci * chunk_rows;
    let rows = cpart.len() / n;
    let tiles = rows / 8;
    for ti in 0..tiles {
        gemm_tile8(i0 + ti * 8, ti * 8, k, n, a, b, bt, cpart);
    }
    for i in tiles * 8..rows {
        gemm_row(i0 + i, i, k, n, a, bt, cpart);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_simd_chunk_avx2(
    ci: usize,
    chunk_rows: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    bt: &[f32],
    cpart: &mut [f32],
) {
    gemm_simd_chunk_impl(ci, chunk_rows, k, n, a, b, bt, cpart);
}

/// 8 rows × 16 columns tile of `c` for rows `[i0, i0 + 8)` (n >= 16).
/// The `S` lanes of each accumulator vector are the tile's COLUMNS —
/// the same structure as a hand-written 8×8 AVX2 kernel: per k-chunk
/// (S k-values) load the 8 `a` row vectors and, for each of the S
/// k-offsets, one `b` row vector (16 contiguous floats at b row
/// `kk + t`, columns `j..j + S` — no transpose needed), then fan each
/// `a` element out over the 16 columns with a `splat` `mul_add`.
/// 8 + S vector loads per chunk of 8×S FMAs; on AVX2 the 16-lane FMAs
/// lower to 2×256-bit.
#[inline(always)]
fn gemm_tile8(
    i0: usize,
    c0: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    bt: &[f32],
    c: &mut [f32],
) {
    let mut j = 0;
    let mut acc = [Simd::<f32, S>::splat(0.0); 8];
    let mut ktail = [0.0f32; 8 * S];
    while j + S <= n {
        for r in 0..8 {
            acc[r] = Simd::<f32, S>::splat(0.0);
        }
        ktail.fill(0.0);
        let mut kk = 0;
        while kk + S <= k {
            let mut a8 = [Simd::<f32, S>::splat(0.0); 8];
            for r in 0..8 {
                // SAFETY: kk + S <= k: S floats at (i0 + r) * k + kk
                // within a's m * k elements.
                a8[r] = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        a.as_ptr().add((i0 + r) * k + kk),
                        S,
                    ))
                };
            }
            let a_arr = a8.map(|v| v.to_array());
            for t in 0..S {
                // SAFETY: j + t < n (j + S <= n, t < S), j + S <= n:
                // S floats at (kk + t) * n + j within b's k * n
                // elements (one b row, contiguous columns).
                let b8 = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        b.as_ptr().add((kk + t) * n + j),
                        S,
                    ))
                };
                for r in 0..8 {
                    acc[r] = Simd::splat(a_arr[r][t]).mul_add(b8, acc[r]);
                }
            }
            kk += S;
        }
        while kk < k {
            for r in 0..8 {
                for t in 0..S {
                    ktail[r * S + t] +=
                        a[(i0 + r) * k + kk] * b[kk * n + j + t];
                }
            }
            kk += 1;
        }
        for r in 0..8 {
            let base = (c0 + r) * n + j;
            for t in 0..S {
                c[base + t] = acc[r].to_array()[t] + ktail[r * S + t];
            }
        }
        j += S;
    }
    for t in j..n {
        gemm_cols_tail(i0, c0, 8, t, k, n, a, bt, c);
    }
}

/// One row of `c` for row `i0` (any n): `n` column accumulators of `S`
/// lanes each (partial sums over k), one `a` row vector per k-chunk
/// fanned out over the columns via per-column `b` vectors (contiguous
/// in the `[n, k]` transpose `bt`), one horizontal reduce per column
/// at the end. Columns are processed in groups of 4 so the `S`-lane
/// accumulators (2×256-bit each on AVX2) fit the register file; any
/// remainder columns use the same per-column SIMD dot.
#[inline(always)]
fn gemm_row(
    i0: usize,
    c0: usize,
    k: usize,
    n: usize,
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
) {
    let mut j = 0;
    while j + 4 <= n {
        let mut acc = [Simd::<f32, S>::splat(0.0); 4];
        let mut ktail = [0.0f32; 4];
        let mut kk = 0;
        while kk + S <= k {
            // SAFETY: kk + S <= k: S floats at kk within arow's k
            // elements.
            let av = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    a.as_ptr().add(i0 * k + kk),
                    S,
                ))
            };
            for t in 0..4 {
                // SAFETY: j + t < n, kk + S <= k: S floats at
                // (j + t) * k + kk within bt's n * k elements.
                let bv = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        bt.as_ptr().add((j + t) * k + kk),
                        S,
                    ))
                };
                acc[t] = av.mul_add(bv, acc[t]);
            }
            kk += S;
        }
        // k-tail: scalar accumulators (adding to the vector would
        // inflate `reduce_sum` by the lane count).
        while kk < k {
            let av = a[i0 * k + kk];
            for t in 0..4 {
                ktail[t] += av * bt[(j + t) * k + kk];
            }
            kk += 1;
        }
        for t in 0..4 {
            c[c0 * n + j + t] = acc[t].reduce_sum() + ktail[t];
        }
        j += 4;
    }
    while j < n {
        gemm_cols_tail(i0, c0, 1, j, k, n, a, bt, c);
        j += 1;
    }
}

/// SIMD dot for the tail columns (any `n`): `c[(c0 + r) * n + t] =
/// dot(a[i0 + r, :], bt[t, :])` for `r in 0..rbase` rows. One S-lane
/// accumulator per row over k-chunks of `S`, so even a single
/// un-aligned column stays vectorized (no scalar fallback).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn gemm_cols_tail(
    i0: usize,
    c0: usize,
    rbase: usize,
    t: usize,
    k: usize,
    n: usize,
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
) {
    for r in 0..rbase {
        let mut acc = Simd::<f32, S>::splat(0.0);
        let mut ktail = 0.0f32;
        let mut kk = 0;
        while kk + S <= k {
            // SAFETY: kk + S <= k: S floats at (i0 + r) * k + kk and
            // t * k + kk within a and bt.
            let av = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    a.as_ptr().add((i0 + r) * k + kk),
                    S,
                ))
            };
            let bv = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    bt.as_ptr().add(t * k + kk),
                    S,
                ))
            };
            acc = av.mul_add(bv, acc);
            kk += S;
        }
        while kk < k {
            ktail += a[(i0 + r) * k + kk] * bt[t * k + kk];
            kk += 1;
        }
        c[(c0 + r) * n + t] = acc.reduce_sum() + ktail;
    }
}

/// LayerNorm over one row: `out[i] = (x[i] - mean) * inv * w[i] + b[i]`
/// with `mean/var` over the row and `inv = 1/sqrt(var/d + eps)`.
/// Mean and variance are computed with one `reduce_sum` per S-lane
/// chunk; the affine pass is two `mul_add`s per chunk.
#[inline(always)]
fn ln_forward_impl(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let d = x.len();
    debug_assert_eq!(w.len(), d);
    debug_assert_eq!(b.len(), d);
    debug_assert_eq!(out.len(), d);
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + S <= d {
        // SAFETY: i + S <= d: S floats at i within x's d elements.
        sum += unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                x.as_ptr().add(i),
                S,
            ))
        }
        .reduce_sum();
        i += S;
    }
    for &v in &x[i..] {
        sum += v;
    }
    let mean = sum / d as f32;
    let mu = Simd::<f32, S>::splat(mean);
    let mut var = 0.0f32;
    let mut i = 0;
    while i + S <= d {
        // SAFETY: i + S <= d: S floats at i within x's d elements.
        let v = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                x.as_ptr().add(i),
                S,
            ))
        };
        let diff = v - mu;
        var += (diff * diff).reduce_sum();
        i += S;
    }
    for &v in &x[i..] {
        let diff = v - mean;
        var += diff * diff;
    }
    let inv = 1.0 / (var / d as f32 + eps).sqrt();
    let mut i = 0;
    while i + S <= d {
        // SAFETY: i + S <= d: S floats at i within each of x, w, b, out.
        let v = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                x.as_ptr().add(i),
                S,
            ))
        };
        let wv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                w.as_ptr().add(i),
                S,
            ))
        };
        let bv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                b.as_ptr().add(i),
                S,
            ))
        };
        let res = (v - mu).mul_add(wv * Simd::<f32, S>::splat(inv), bv);
        out[i..i + S].copy_from_slice(&res.to_array());
        i += S;
    }
    for (ii, &v) in x[i..].iter().enumerate() {
        let idx = i + ii;
        out[idx] = (v - mean) * inv * w[idx] + b[idx];
    }
}

dispatch_avx2!(
    ln_forward,
    ln_forward_impl,
    ln_forward_avx2,
    (),
    [x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]]
);

/// Silu (swish) in place: `v = v * sigmoid(v)`.
#[inline(always)]
fn silu_into_impl(v: &mut [f32]) {
    let mut i = 0;
    while i + S <= v.len() {
        // SAFETY: i + S <= v.len(): S floats at i within v.
        let x = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                v.as_ptr().add(i),
                S,
            ))
        };
        let res = x.mul_add(
            (Simd::<f32, S>::splat(1.0) + (-x).exp()).recip(),
            Simd::<f32, S>::splat(0.0),
        );
        v[i..i + S].copy_from_slice(&res.to_array());
        i += S;
    }
    for x in &mut v[i..] {
        *x = *x * (1.0 + (-*x).exp()).recip();
    }
}

dispatch_avx2!(silu_into, silu_into_impl, silu_into_avx2, (), [v: &mut [f32]]);

/// ReLU in place.
#[inline(always)]
fn relu_into_impl(v: &mut [f32]) {
    let zero = Simd::<f32, S>::splat(0.0);
    let mut i = 0;
    while i + S <= v.len() {
        // SAFETY: i + S <= v.len(): S floats at i within v.
        let x = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                v.as_ptr().add(i),
                S,
            ))
        };
        let res = x.simd_max(zero);
        v[i..i + S].copy_from_slice(&res.to_array());
        i += S;
    }
    for x in &mut v[i..] {
        *x = x.max(0.0);
    }
}

dispatch_avx2!(relu_into, relu_into_impl, relu_into_avx2, (), [v: &mut [f32]]);

/// GLU from an interleaved `[t, 2d]` buffer into `[t, d]`:
/// `out[tt * d + i] = h[tt * 2d + i] * sigmoid(h[tt * 2d + d + i])`.
#[inline(always)]
fn glu_from_impl(h: &[f32], d: usize, out: &mut [f32]) {
    let t = h.len() / (2 * d);
    debug_assert_eq!(out.len(), t * d);
    for tt in 0..t {
        let base = tt * 2 * d;
        let obase = tt * d;
        let mut i = 0;
        while i + S <= d {
            // SAFETY: i + S <= d: S floats at base + i and
            // base + d + i within h's t * 2d elements.
            let gate = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    h.as_ptr().add(base + i),
                    S,
                ))
            };
            let val = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    h.as_ptr().add(base + d + i),
                    S,
                ))
            };
            let res = gate.mul_add(
                (Simd::<f32, S>::splat(1.0) + (-val).exp()).recip(),
                Simd::<f32, S>::splat(0.0),
            );
            out[obase + i..obase + i + S].copy_from_slice(&res.to_array());
            i += S;
        }
        for i in i..d {
            out[obase + i] =
                h[base + i] * (1.0 + (-h[base + d + i]).exp()).recip();
        }
    }
}

dispatch_avx2!(
    glu_from,
    glu_from_impl,
    glu_from_avx2,
    (),
    [h: &[f32], d: usize, out: &mut [f32]]
);

/// One attention score for a `(query, head, key)` triple:
/// `scale * sum_i ((qu[i] + uh[i]) * kk[i] + (qv[i] + vh[i]) * p[i])`
/// over `head_dim` lanes. The `q_u`- and `q_v`-offset parts are loaded
/// once per (query, head) by the caller (`qu`/`qv`), so the inner loop
/// is two `mul_add`s per S-lane chunk.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn score_dot_impl(
    qu: &[f32],
    uh: &[f32],
    qv: &[f32],
    vh: &[f32],
    kk: &[f32],
    p: &[f32],
    scale: f32,
) -> f32 {
    let hd = qu.len();
    debug_assert_eq!(uh.len(), hd);
    debug_assert_eq!(qv.len(), hd);
    debug_assert_eq!(vh.len(), hd);
    debug_assert_eq!(kk.len(), hd);
    debug_assert_eq!(p.len(), hd);
    let mut acc = Simd::<f32, S>::splat(0.0);
    let mut i = 0;
    while i + S <= hd {
        // SAFETY: i + S <= hd: S floats at i within each of the six
        // head_dim-length slices.
        let quv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                qu.as_ptr().add(i),
                S,
            ))
        };
        let uhv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                uh.as_ptr().add(i),
                S,
            ))
        };
        let qvv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                qv.as_ptr().add(i),
                S,
            ))
        };
        let vhv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                vh.as_ptr().add(i),
                S,
            ))
        };
        let kkv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                kk.as_ptr().add(i),
                S,
            ))
        };
        let pv = unsafe {
            Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                p.as_ptr().add(i),
                S,
            ))
        };
        acc = (quv + uhv).mul_add(kkv, (qvv + vhv).mul_add(pv, acc));
        i += S;
    }
    let mut sum = acc.reduce_sum();
    for i in i..hd {
        sum += (qu[i] + uh[i]) * kk[i] + (qv[i] + vh[i]) * p[i];
    }
    sum * scale
}

dispatch_avx2!(
    score_dot,
    score_dot_impl,
    score_dot_avx2,
    f32,
    [
        qu: &[f32],
        uh: &[f32],
        qv: &[f32],
        vh: &[f32],
        kk: &[f32],
        p: &[f32],
        scale: f32
    ]
);

/// Softmax over the attention band `[k0, k1)` for one head `h`, then
/// fold the weights into the `v` rows:
/// `out[i] = sum_kk exp(srow[kk * n_heads + h] - max) / sum * vv[kk, i]`
/// for `i in 0..head_dim`, where `v_at(kk)` is the head's `v` row
/// slice (contiguous `head_dim` floats). The caller passes the full
/// per-query row of `scores` (strided by `n_heads`) and the band
/// indices in its own frame (`k0..k1`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn softmax_v_impl<'a, F>(
    srow: &[f32],
    k0: usize,
    k1: usize,
    n_heads: usize,
    h: usize,
    v_at: F,
    head_dim: usize,
    out: &mut [f32],
) where
    F: Fn(usize) -> &'a [f32],
{
    let mut maxv = f32::NEG_INFINITY;
    for kk in k0..k1 {
        maxv = maxv.max(srow[kk * n_heads + h]);
    }
    let band = k1 - k0;
    // Cache `exp(score - max)` once per band entry: recomputing it in
    // the inner `head_dim` loop would turn a multiply into a
    // transcendental per (band, head_dim) pair. The hot path (band up
    // to 64) uses a fixed stack cache; larger bands fall back to a Vec
    // allocation, which is cheaper than the repeated exp() and happens
    // only as the streaming window grows.
    if band <= 64 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut e = [0.0f32; 64];
        for kk in k0..k1 {
            let val = (srow[kk * n_heads + h] - maxv).exp();
            e[i] = val;
            sum += val;
            i += 1;
        }
        let inv = 1.0 / sum;
        let mut j = 0;
        while j + S <= head_dim {
            let mut acc = Simd::<f32, S>::splat(0.0);
            for (ii, kk) in (k0..k1).enumerate() {
                let vv = v_at(kk);
                // SAFETY: j + S <= head_dim: S floats at j within vv (the
                // caller guarantees vv has head_dim elements).
                let v = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        vv.as_ptr().add(j),
                        S,
                    ))
                };
                acc = Simd::<f32, S>::splat(e[ii] * inv).mul_add(v, acc);
            }
            out[j..j + S].copy_from_slice(&acc.to_array());
            j += S;
        }
        for i in j..head_dim {
            let mut acc = 0.0f32;
            for (ii, kk) in (k0..k1).enumerate() {
                acc += e[ii] * inv * v_at(kk)[i];
            }
            out[i] = acc;
        }
    } else {
        let mut e = vec![0.0f32; band];
        let mut sum = 0.0f32;
        for (i, kk) in (k0..k1).enumerate() {
            let val = (srow[kk * n_heads + h] - maxv).exp();
            e[i] = val;
            sum += val;
        }
        let inv = 1.0 / sum;
        let mut j = 0;
        while j + S <= head_dim {
            let mut acc = Simd::<f32, S>::splat(0.0);
            for (ii, kk) in (k0..k1).enumerate() {
                let vv = v_at(kk);
                // SAFETY: j + S <= head_dim: S floats at j within vv (the
                // caller guarantees vv has head_dim elements).
                let v = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        vv.as_ptr().add(j),
                        S,
                    ))
                };
                acc = Simd::<f32, S>::splat(e[ii] * inv).mul_add(v, acc);
            }
            out[j..j + S].copy_from_slice(&acc.to_array());
            j += S;
        }
        for i in j..head_dim {
            let mut acc = 0.0f32;
            for (ii, kk) in (k0..k1).enumerate() {
                acc += e[ii] * inv * v_at(kk)[i];
            }
            out[i] = acc;
        }
    }
}

/// Runtime dispatch for [`softmax_v_impl`]: on x86_64 with runtime-verified
/// AVX2+FMA the `#[target_feature]` wrapper (which inlines the impl and
/// thus compiles its `std::simd` ops with avx2 codegen) is used, otherwise
/// the plain impl. Written by hand (not `dispatch_avx2!`) because the
/// closure type is generic over a lifetime.
#[allow(clippy::too_many_arguments)]
pub fn softmax_v<'a, F>(
    srow: &[f32],
    k0: usize,
    k1: usize,
    n_heads: usize,
    h: usize,
    v_at: F,
    head_dim: usize,
    out: &mut [f32],
) where
    F: Fn(usize) -> &'a [f32],
{
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the avx2+fma feature set was runtime-verified above.
            return unsafe {
                softmax_v_avx2(srow, k0, k1, n_heads, h, v_at, head_dim, out)
            };
        }
    }
    softmax_v_impl(srow, k0, k1, n_heads, h, v_at, head_dim, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn softmax_v_avx2<'a, F>(
    srow: &[f32],
    k0: usize,
    k1: usize,
    n_heads: usize,
    h: usize,
    v_at: F,
    head_dim: usize,
    out: &mut [f32],
) where
    F: Fn(usize) -> &'a [f32],
{
    softmax_v_impl(srow, k0, k1, n_heads, h, v_at, head_dim, out);
}

/// 1D depthwise conv: `out[ot * dim + c] = sum_k x[(ot - pad_left + k)
/// * dim + c] * w[c * kh + k]` for `ot in 0..t_out`, `t_out =
/// t + pad_left + pad_right - kh + 1`. The `w` rows (one per channel)
/// are loaded once per (ot, channel-chunk) — the `kh`-length dot is
/// one S-lane `mul_add` chain when `kh <= S`.
#[inline(always)]
fn dwconv_forward_impl(
    x: &[f32],
    t: usize,
    dim: usize,
    kh: usize,
    pad_left: usize,
    pad_right: usize,
    w: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(w.len(), dim * kh);
    let t_out = t + pad_left + pad_right - kh + 1;
    debug_assert_eq!(out.len(), t_out * dim);
    // The per-channel weight row is `kh` floats; the k-loop is a
    // scalar gather (kh <= S) into an S-lane accumulator per channel.
    for ot in 0..t_out {
        let t0 = ot as isize - pad_left as isize;
        let orow = &mut out[ot * dim..(ot + 1) * dim];
        let mut c = 0;
        while c + S <= dim {
            let mut acc = Simd::<f32, S>::splat(0.0);
            for k in 0..kh {
                let ti = t0 + k as isize;
                if ti < 0 || ti as usize >= t {
                    continue;
                }
                // SAFETY: c + S <= dim: S floats at ti * dim + c within
                // x's t * dim elements.
                let xv = unsafe {
                    Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                        x.as_ptr().add(ti as usize * dim + c),
                        S,
                    ))
                };
                // The S channels' weights at tap k are strided by kh,
                // so gather them per lane.
                let mut wv = [0.0f32; S];
                for l in 0..S {
                    wv[l] = w[(c + l) * kh + k];
                }
                let wv = Simd::<f32, S>::from_array(wv);
                acc = xv.mul_add(wv, acc);
            }
            orow[c..c + S].copy_from_slice(&acc.to_array());
            c += S;
        }
        for c in c..dim {
            let mut acc = 0.0f32;
            for k in 0..kh {
                let ti = t0 + k as isize;
                if ti < 0 || ti as usize >= t {
                    continue;
                }
                acc += x[ti as usize * dim + c] * w[c * kh + k];
            }
            orow[c] = acc;
        }
    }
}

dispatch_avx2!(
    dwconv_forward,
    dwconv_forward_impl,
    dwconv_forward_avx2,
    (),
    [
        x: &[f32],
        t: usize,
        dim: usize,
        kh: usize,
        pad_left: usize,
        pad_right: usize,
        w: &[f32],
        out: &mut [f32]
    ]
);

/// `y[j] = dot(f[j, :], x) + bias[j]` for `j in 0..out` (a row-major
/// `[out, inp]` weight matrix against a single input row).
#[inline(always)]
fn f32_matvec_impl(
    f: &[f32],
    inp: usize,
    out: usize,
    x: &[f32],
    y: &mut [f32],
) {
    debug_assert_eq!(f.len(), out * inp);
    debug_assert_eq!(x.len(), inp);
    for j in 0..out {
        let row = &f[j * inp..(j + 1) * inp];
        let mut acc = Simd::<f32, S>::splat(0.0);
        let mut i = 0;
        while i + S <= inp {
            // SAFETY: i + S <= inp: S floats at i within row and x.
            let rv = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    row.as_ptr().add(i),
                    S,
                ))
            };
            let xv = unsafe {
                Simd::<f32, S>::from_slice(std::slice::from_raw_parts(
                    x.as_ptr().add(i),
                    S,
                ))
            };
            acc = rv.mul_add(xv, acc);
            i += S;
        }
        let mut sum = acc.reduce_sum();
        for i in i..inp {
            sum += row[i] * x[i];
        }
        y[j] = sum;
    }
}

dispatch_avx2!(
    f32_matvec,
    f32_matvec_impl,
    f32_matvec_avx2,
    (),
    [f: &[f32], inp: usize, out: usize, x: &[f32], y: &mut [f32]]
);

/// Transpose `[rows, cols]` row-major into `[cols, rows]` row-major:
/// `out[c * rows + r] = x[r * cols + c]`. Each output column chunk of
/// `S` rows is gathered into an S-lane vector (the source is strided by
/// `cols`) and stored contiguously.
#[inline(always)]
fn transpose_simd_impl(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(out.len(), cols * rows);
    for c in 0..cols {
        let obase = c * rows;
        let mut r = 0;
        while r + S <= rows {
            let mut arr = [0.0f32; S];
            for i in 0..S {
                arr[i] = x[(r + i) * cols + c];
            }
            let v = Simd::<f32, S>::from_array(arr);
            out[obase + r..obase + r + S].copy_from_slice(&v.to_array());
            r += S;
        }
        for r in r..rows {
            out[obase + r] = x[r * cols + c];
        }
    }
}

dispatch_avx2!(
    transpose_simd,
    transpose_simd_impl,
    transpose_simd_avx2,
    (),
    [x: &[f32], rows: usize, cols: usize, out: &mut [f32]]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgemm_kernel::q8_gemm_scalar;

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
        q8_gemm_simd(m, k, n, &w, padded_row, block_bytes, 2, &x, &mut y);
        q8_gemm_scalar(m, k, n, &w, padded_row, block_bytes, 2, &x, &mut y_ref);
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

    #[test]
    fn simd_matches_scalar() {
        run_case(8, 64, 1);
        run_case(16, 128, 4);
        run_case(64, 512, 8);
        run_case(128, 2048, 16);
        run_case(512, 2048, 16);
    }

    fn gemm_run_case(m: usize, k: usize, n: usize) {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 3 + 1) % 97) as f32 / 13.0 - 1.0;
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i * 5 + 2) % 89) as f32 / 11.0 - 1.0;
        }
        let mut y = vec![0.0f32; m * n];
        let mut y_ref = vec![0.0f32; m * n];
        gemm_simd_into(m, k, n, &a, &b, &mut y);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                y_ref[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            let d = (y[i] - y_ref[i]).abs();
            let tol = 1e-3 * (1.0 + y_ref[i].abs());
            assert!(
                d <= tol,
                "gemm m={m} k={k} n={n} [{i}]: {} vs {}",
                y[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn gemm_matches_scalar() {
        gemm_run_case(8, 8, 8);
        gemm_run_case(16, 32, 4);
        gemm_run_case(64, 512, 16);
        gemm_run_case(64, 512, 7);
        gemm_run_case(128, 768, 14);
        gemm_run_case(64, 500, 16);
        gemm_run_case(64, 512, 20);
        gemm_run_case(7, 512, 16);
        gemm_run_case(512, 512, 16);
        gemm_run_case(768, 1000, 128);
    }

    fn ln_run_case(dim: usize) {
        let x: Vec<f32> = (0..dim)
            .map(|i| ((i * 7) % 113) as f32 / 5.0 - 3.0)
            .collect();
        let w: Vec<f32> =
            (0..dim).map(|i| 1.0 + ((i % 9) as f32) / 17.0).collect();
        let b: Vec<f32> =
            (0..dim).map(|i| ((i % 5) as f32 - 2.0) / 3.0).collect();
        let eps = 1e-5;
        let mut y = vec![0.0f32; dim];
        let mut y_ref = vec![0.0f32; dim];
        ln_forward(&x, &w, &b, eps, &mut y);
        let mean = x.iter().sum::<f32>() / dim as f32;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>()
            / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            y_ref[i] = (x[i] - mean) * inv * w[i] + b[i];
        }
        for i in 0..dim {
            let d = (y[i] - y_ref[i]).abs();
            let tol = 1e-3 * (1.0 + y_ref[i].abs());
            assert!(d <= tol, "ln dim={dim} [{i}]: {} vs {}", y[i], y_ref[i]);
        }
    }

    fn ops_run_case(dim: usize) {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| ((i * 3) % 101) as f32 / 7.0 - 4.0)
            .collect();
        let v_ref = v.clone();
        silu_into(&mut v);
        for i in 0..dim {
            let r = v_ref[i] * (1.0 + (-v_ref[i]).exp()).recip();
            let d = (v[i] - r).abs();
            let tol = 1e-4 * (1.0 + r.abs());
            assert!(d <= tol, "silu dim={dim} [{i}]: {} vs {}", v[i], r);
        }
        relu_into(&mut v);
        for i in 0..dim {
            let r = v_ref[i] * (1.0 + (-v_ref[i]).exp()).recip();
            assert!(
                (v[i] - r.max(0.0)).abs() <= 1e-4 * (1.0 + r.abs()),
                "relu"
            );
        }
    }

    fn glu_run_case(t: usize, d: usize) {
        let h: Vec<f32> = (0..t * 2 * d)
            .map(|i| ((i * 11) % 97) as f32 / 9.0 - 2.0)
            .collect();
        let mut out = vec![0.0f32; t * d];
        let mut out_ref = vec![0.0f32; t * d];
        glu_from(&h, d, &mut out);
        for tt in 0..t {
            for i in 0..d {
                let gate = h[tt * 2 * d + i];
                let val = h[tt * 2 * d + d + i];
                out_ref[tt * d + i] = gate * (1.0 + (-val).exp()).recip();
            }
        }
        for i in 0..t * d {
            let d = (out[i] - out_ref[i]).abs();
            let tol = 1e-4 * (1.0 + out_ref[i].abs());
            assert!(d <= tol, "glu [{i}]: {} vs {}", out[i], out_ref[i]);
        }
    }

    fn score_run_case(head_dim: usize) {
        let qu: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 3) % 101) as f32 / 11.0 - 1.0)
            .collect();
        let uh: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 5) % 97) as f32 / 13.0 - 2.0)
            .collect();
        let qv: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 7) % 89) as f32 / 17.0 - 1.5)
            .collect();
        let vh: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 9) % 83) as f32 / 19.0 - 0.5)
            .collect();
        let kk_d: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 13) % 79) as f32 / 23.0 - 1.0)
            .collect();
        let pos_row: Vec<f32> = (0..head_dim)
            .map(|i| ((i * 17) % 73) as f32 / 29.0 - 2.0)
            .collect();
        let scale = 0.125;
        let got = score_dot(&qu, &uh, &qv, &vh, &kk_d, &pos_row, scale);
        let mut acc = 0.0f32;
        for i in 0..head_dim {
            let qui = qu[i] + uh[i];
            let qvi = qv[i] + vh[i];
            acc += qui * kk_d[i] + qvi * pos_row[i];
        }
        let want = acc * scale;
        assert!(
            (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "score: {} vs {}",
            got,
            want
        );
    }

    fn softmax_v_run_case(band: usize, head_dim: usize, n_heads: usize) {
        let k1 = band;
        let k0 = band / 3;
        let h = (band / 2) % n_heads;
        let mut srow = vec![0.0f32; k1 * n_heads];
        let mut v = vec![0.0f32; band * head_dim];
        for i in 0..srow.len() {
            srow[i] = ((i * 19) % 71) as f32 / 31.0 - 1.0;
        }
        for i in 0..v.len() {
            v[i] = ((i * 23) % 67) as f32 / 37.0 - 0.5;
        }
        let mut out = vec![0.0f32; head_dim];
        softmax_v(
            &srow,
            k0,
            k1,
            n_heads,
            h,
            |i| &v[i * head_dim..(i + 1) * head_dim],
            head_dim,
            &mut out,
        );
        let mut maxv = f32::NEG_INFINITY;
        for kk in k0..k1 {
            maxv = maxv.max(srow[kk * n_heads + h]);
        }
        let mut sum = 0.0f32;
        for kk in k0..k1 {
            sum += (srow[kk * n_heads + h] - maxv).exp();
        }
        let inv = 1.0 / sum;
        for j in 0..head_dim {
            let mut acc = 0.0f32;
            for kk in k0..k1 {
                acc += (srow[kk * n_heads + h] - maxv).exp()
                    * inv
                    * v[kk * head_dim + j];
            }
            let tol = 1e-3 * (1.0 + acc.abs());
            assert!(
                (out[j] - acc).abs() <= tol,
                "softmax [{j}]: {} vs {}",
                out[j],
                acc
            );
        }
    }

    fn dwconv_run_case(
        t: usize,
        dim: usize,
        kh: usize,
        pad_left: usize,
        pad_right: usize,
    ) {
        let x: Vec<f32> = (0..t * dim)
            .map(|i| ((i * 5) % 103) as f32 / 21.0 - 1.0)
            .collect();
        let w: Vec<f32> = (0..dim * kh)
            .map(|i| ((i * 7) % 59) as f32 / 33.0 - 0.5)
            .collect();
        let t_out = t + pad_left + pad_right - kh + 1;
        let mut out = vec![0.0f32; t_out * dim];
        let mut out_ref = vec![0.0f32; t_out * dim];
        dwconv_forward(&x, t, dim, kh, pad_left, pad_right, &w, &mut out);
        for ot in 0..t_out {
            let t0 = ot as isize - pad_left as isize;
            for c in 0..dim {
                let mut acc = 0.0f32;
                for k in 0..kh {
                    let ti = t0 + k as isize;
                    if ti < 0 || ti as usize >= t {
                        continue;
                    }
                    acc += x[ti as usize * dim + c] * w[c * kh + k];
                }
                out_ref[ot * dim + c] = acc;
            }
        }
        for i in 0..out.len() {
            let d = (out[i] - out_ref[i]).abs();
            let tol = 1e-3 * (1.0 + out_ref[i].abs());
            assert!(d <= tol, "dwconv [{i}]: {} vs {}", out[i], out_ref[i]);
        }
    }

    fn matvec_run_case(inp: usize, out: usize) {
        let f: Vec<f32> = (0..inp * out)
            .map(|i| ((i * 3) % 97) as f32 / 15.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..inp)
            .map(|i| ((i * 5) % 91) as f32 / 19.0 - 0.5)
            .collect();
        let mut y = vec![0.0f32; out];
        let mut y_ref = vec![0.0f32; out];
        f32_matvec(&f, inp, out, &x, &mut y);
        for j in 0..out {
            let row = &f[j * inp..(j + 1) * inp];
            let mut acc = 0.0f32;
            for i in 0..inp {
                acc += row[i] * x[i];
            }
            y_ref[j] = acc;
        }
        for j in 0..out {
            let d = (y[j] - y_ref[j]).abs();
            let tol = 1e-3 * (1.0 + y_ref[j].abs());
            assert!(d <= tol, "matvec [{j}]: {} vs {}", y[j], y_ref[j]);
        }
    }

    fn transpose_run_case(rows: usize, cols: usize) {
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 109) as f32 / 25.0 - 1.0)
            .collect();
        let mut y = vec![0.0f32; rows * cols];
        transpose_simd(&x, rows, cols, &mut y);
        for c in 0..cols {
            for r in 0..rows {
                assert!(
                    (y[c * rows + r] - x[r * cols + c]).abs() <= 1e-5,
                    "transpose [{r},{c}]"
                );
            }
        }
    }

    #[test]
    fn helpers_match_scalar() {
        ln_run_case(64);
        ln_run_case(512);
        ln_run_case(37);
        ops_run_case(64);
        ops_run_case(512);
        ops_run_case(33);
        glu_run_case(4, 64);
        glu_run_case(16, 512);
        score_run_case(64);
        score_run_case(512);
        softmax_v_run_case(16, 64, 8);
        softmax_v_run_case(61, 64, 8);
        softmax_v_run_case(96, 64, 8);
        softmax_v_run_case(128, 512, 8);
        dwconv_run_case(32, 64, 3, 14, 0);
        dwconv_run_case(64, 512, 31, 14, 0);
        matvec_run_case(64, 128);
        matvec_run_case(768, 256);
        transpose_run_case(16, 64);
        transpose_run_case(64, 512);
        transpose_run_case(7, 33);
    }
}
