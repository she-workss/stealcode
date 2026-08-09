//! Quick numerical check: q8_gemm (int8) vs a scalar int8 reference
//! and the dequantized f32 gemm.
//! Usage: cargo run -r -p voice --example q8test

use std::time::Instant;

use voice::nemotron::sgemm_kernel::q8_gemm;

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if man == 0 {
                sign << 31
            } else {
                let mut e = 127 - 15 + 1;
                let mut m = (man << 13) as u32;
                while m & 0x0080_0000 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x007f_ffff;
                (sign << 31) | ((e as u32) << 23) | m
            }
        }
        0x1f => (sign << 31) | 0x7fc0_0000 | (man << 13),
        e => (sign << 31) | ((e + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7fffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 0x1f {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let man16 = (man | 0x800000) >> (14 - exp16);
        return sign | man16 as u16;
    }
    sign | ((exp16 as u16) << 10) | ((man >> 13) as u16)
}

fn scale_of(w: &[u8], base: usize, bb: usize) -> f32 {
    if bb == 34 {
        f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]))
    } else {
        f32::from_le_bytes([w[base], w[base + 1], w[base + 2], w[base + 3]])
    }
}

/// Scalar int8 reference: quantize activations per block, dot in int8,
/// rescale per block. Matches the intended `q8_gemm` math exactly.
fn q8_ref(
    m: usize,
    k: usize,
    n: usize,
    w: &[u8],
    padded_row: usize,
    bb: usize,
    qoff: usize,
    x: &[f32],
) -> Vec<f32> {
    let nblocks = k.div_ceil(32);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for nj in 0..n {
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let k0 = b * 32;
                let len = 32usize.min(k - k0);
                let wbase = i * padded_row + b * bb;
                let dw = scale_of(w, wbase, bb);
                let mut maxv = 0.0f32;
                for j in 0..len {
                    maxv = maxv.max(x[(k0 + j) * n + nj].abs());
                }
                let s = if maxv == 0.0 { 1.0 } else { maxv / 127.0 };
                if maxv == 0.0 {
                    continue;
                }
                let inv = 1.0 / s;
                let mut dot = 0i64;
                for j in 0..len {
                    let q = (x[(k0 + j) * n + nj] * inv).round() as i8;
                    dot += (w[wbase + qoff + j] as i8 as i64) * (q as i64);
                }
                acc += (dw * s) * dot as f32;
            }
            out[i * n + nj] = acc;
        }
    }
    out
}

/// Dequantize weights to f32 then do a plain dot-product reference.
fn f32_ref(
    m: usize,
    k: usize,
    n: usize,
    w: &[u8],
    padded_row: usize,
    bb: usize,
    qoff: usize,
    x: &[f32],
) -> Vec<f32> {
    let nblocks = k.div_ceil(32);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for nj in 0..n {
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let k0 = b * 32;
                let len = 32usize.min(k - k0);
                let wbase = i * padded_row + b * bb;
                let dw = scale_of(w, wbase, bb);
                for j in 0..len {
                    let wv = dw * (w[wbase + qoff + j] as i8 as f32);
                    acc += wv * x[(k0 + j) * n + nj];
                }
            }
            out[i * n + nj] = acc;
        }
    }
    out
}

fn test(m: usize, k: usize, n: usize, bb: usize, qoff: usize) {
    let nblocks = k.div_ceil(32);
    let padded_row = nblocks * bb;
    let mut w = vec![0u8; m * padded_row];
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    for i in 0..m {
        for b in 0..nblocks {
            let base = i * padded_row + b * bb;
            let d = (((rng & 0xffff) as f32 / 32768.0) - 1.0) * 0.1;
            if bb == 34 {
                let f16 = f32_to_f16(d);
                w[base] = f16 as u8;
                w[base + 1] = (f16 >> 8) as u8;
            } else {
                w[base..base + 4].copy_from_slice(&d.to_le_bytes());
            }
            for j in 0..32 {
                let v = (rng >> 7) as i8;
                w[base + qoff + j] = v as u8;
            }
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
    }
    let mut x = vec![0.0f32; k * n];
    let mut rng2: u64 = 0xdead_beef_cafe_f00d;
    for v in &mut x {
        *v = ((rng2 & 0xffff) as f32 / 32768.0) * 2.0 - 1.0;
        rng2 = rng2.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    let s0 = scale_of(&w, 0, bb);
    let s_last = scale_of(&w, (m - 1) * padded_row + (nblocks - 1) * bb, bb);
    println!(
        "  dbg: bb={} d0={:.6} dlast={:.6} i80={}",
        bb, s0, s_last, w[qoff] as i8
    );
    let ref_i8 = q8_ref(m, k, n, &w, padded_row, bb, qoff, &x);
    let ref_f32 = f32_ref(m, k, n, &w, padded_row, bb, qoff, &x);
    let mut c_q = vec![0.0f32; m * n];
    q8_gemm(m, k, n, &w, padded_row, bb, qoff, &x, &mut c_q);
    let mut maxdiff = 0.0f32;
    for i in 0..m * n {
        maxdiff = maxdiff.max((c_q[i] - ref_i8[i]).abs());
    }
    let mut diff_f32 = 0.0f32;
    for i in 0..m * n {
        diff_f32 = diff_f32.max((c_q[i] - ref_f32[i]).abs());
    }
    let ref_max = ref_f32.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!(
        "{}x{} n={} bb={}: avx2-vs-scalar={:.5} avx2-vs-f32={:.4} rel={:.4} (ref_max={:.4})",
        m,
        k,
        n,
        bb,
        maxdiff,
        diff_f32,
        diff_f32 / ref_max,
        ref_max
    );

    let mut t0 = Instant::now();
    for _ in 0..10 {
        q8_gemm(m, k, n, &w, padded_row, bb, qoff, &x, &mut c_q);
    }
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    let flop = 2.0 * m as f64 * k as f64 * n as f64;
    println!(
        "  q8_gemm: {:.3} ms ({:.0} GF/s)",
        dt * 1e3,
        flop / dt / 1e9
    );
    std::hint::black_box(&c_q);
}

fn main() {
    test(4096, 1024, 32, 34, 2);
    test(1024, 4096, 32, 34, 2);
    test(768, 768, 32, 34, 2);
    test(13088, 640, 1, 34, 2);
    test(1024, 1024, 16, 36, 4);
    test(2048, 1024, 81, 34, 2);
    test(4096, 1024, 26, 34, 2);
}
