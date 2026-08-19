//! Criterion benchmarks of the portable SIMD elementwise / attention /
//! conv / matvec helpers vs their scalar references, at the streaming
//! encoder shapes.
//!
//!   cargo bench -p audio_engine --bench bench_ops
//!
//! Kernel order within a group is fixed (simd first, then scalar): on
//! this thermally-drifting machine the first benches run on the
//! freshest silicon, so a stable documented order keeps the
//! comparisons comparable across runs.

use std::hint::black_box;

use audio_engine::simd_kernel::{
    dwconv_forward, f32_matvec, glu_from, ln_forward, score_dot, silu_into,
    softmax_v, transpose_simd,
};
use criterion::{
    BatchSize, Criterion, Throughput, criterion_group, criterion_main,
};

const D: usize = 512;
const HEAD_DIM: usize = 64;
const N_HEADS: usize = 8;
const BAND: usize = 61;

fn make(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i * 3 + 1) % 97) as f32 / 13.0 - 1.0)
        .collect()
}

fn bench_ln(c: &mut Criterion) {
    let x = make(D);
    let w = make(D);
    let b = make(D);
    let mut group = c.benchmark_group("ln_512");
    group.throughput(Throughput::Elements(D as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; D],
            |mut out| {
                ln_forward(
                    black_box(&x),
                    black_box(&w),
                    black_box(&b),
                    1e-5,
                    &mut out,
                );
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; D],
            |mut out| {
                let mean = x.iter().sum::<f32>() / D as f32;
                let var =
                    x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>()
                        / D as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for i in 0..D {
                    out[i] = (x[i] - mean) * inv * w[i] + b[i];
                }
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_silu(c: &mut Criterion) {
    let mut v = make(D * 4);
    let v_ref = v.clone();
    let mut group = c.benchmark_group("silu_2048");
    group.throughput(Throughput::Elements(v.len() as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || v_ref.clone(),
            |mut v| silu_into(black_box(&mut v)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || v_ref.clone(),
            |mut v| {
                for x in v.iter_mut() {
                    *x = *x * (1.0 + (-*x).exp()).recip();
                }
            },
            BatchSize::SmallInput,
        )
    });
    black_box(&mut v);
    group.finish();
}

fn bench_glu(c: &mut Criterion) {
    let h = make(4 * 2 * D);
    let mut group = c.benchmark_group("glu_4x512");
    group.throughput(Throughput::Elements((4 * D) as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; 4 * D],
            |mut out| {
                glu_from(black_box(&h), D, &mut out);
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; 4 * D],
            |mut out| {
                for tt in 0..4 {
                    for i in 0..D {
                        let gate = h[tt * 2 * D + i];
                        let val = h[tt * 2 * D + D + i];
                        out[tt * D + i] = gate * (1.0 + (-val).exp()).recip();
                    }
                }
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let qu = make(HEAD_DIM);
    let uh = make(HEAD_DIM);
    let qv = make(HEAD_DIM);
    let vh = make(HEAD_DIM);
    let kk = make(HEAD_DIM);
    let pos = make(HEAD_DIM);
    let mut group = c.benchmark_group("score_64");
    group.throughput(Throughput::Elements(HEAD_DIM as u64));
    group.bench_function("simd", |be| {
        be.iter(|| {
            score_dot(
                black_box(&qu),
                black_box(&uh),
                black_box(&qv),
                black_box(&vh),
                black_box(&kk),
                black_box(&pos),
                0.125,
            )
        })
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || (),
            |_| {
                let mut acc = 0.0f32;
                for i in 0..HEAD_DIM {
                    let qui = qu[i] + uh[i];
                    let qvi = qv[i] + vh[i];
                    acc += qui * kk[i] + qvi * pos[i];
                }
                black_box(acc * 0.125);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let srow: Vec<f32> = (0..BAND * N_HEADS)
        .map(|i| ((i * 19) % 71) as f32 / 31.0 - 1.0)
        .collect();
    let v: Vec<f32> = (0..BAND * HEAD_DIM)
        .map(|i| ((i * 23) % 67) as f32 / 37.0 - 0.5)
        .collect();
    let mut out = vec![0.0f32; HEAD_DIM];
    let v_at = |i: usize| &v[i * HEAD_DIM..(i + 1) * HEAD_DIM];
    let mut group = c.benchmark_group("softmax_61x64");
    group.throughput(Throughput::Elements((BAND * HEAD_DIM) as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; HEAD_DIM],
            |mut out| {
                softmax_v(
                    black_box(&srow),
                    0,
                    BAND,
                    N_HEADS,
                    0,
                    &v_at,
                    HEAD_DIM,
                    &mut out,
                );
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; HEAD_DIM],
            |mut out| {
                let mut maxv = f32::NEG_INFINITY;
                for kk in 0..BAND {
                    maxv = maxv.max(srow[kk]);
                }
                let mut sum = 0.0f32;
                for kk in 0..BAND {
                    sum += (srow[kk] - maxv).exp();
                }
                let inv = 1.0 / sum;
                for j in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    for kk in 0..BAND {
                        acc += (srow[kk] - maxv).exp()
                            * inv
                            * v[kk * HEAD_DIM + j];
                    }
                    out[j] = acc;
                }
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_dwconv(c: &mut Criterion) {
    let t = 64;
    let kh = 31;
    let x = make(t * D);
    let w = make(D * kh);
    let t_out = t + 14 - kh + 1;
    let mut group = c.benchmark_group("dwconv_64x512_k31");
    group.throughput(Throughput::Elements((t * D) as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; t_out * D],
            |mut out| {
                dwconv_forward(
                    black_box(&x),
                    t,
                    D,
                    kh,
                    14,
                    0,
                    black_box(&w),
                    &mut out,
                );
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; t_out * D],
            |mut out| {
                for ot in 0..t_out {
                    let t0 = ot as isize - 14;
                    for c in 0..D {
                        let mut acc = 0.0f32;
                        for k in 0..kh {
                            let ti = t0 + k as isize;
                            if ti < 0 || ti as usize >= t {
                                continue;
                            }
                            acc += x[ti as usize * D + c] * w[c * kh + k];
                        }
                        out[ot * D + c] = acc;
                    }
                }
                black_box(out.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let inp = 768;
    let out = 256;
    let f = make(inp * out);
    let x = make(inp);
    let mut y = vec![0.0f32; out];
    let mut group = c.benchmark_group("matvec_768x256");
    group.throughput(Throughput::Elements((inp * out) as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; out],
            |mut y| {
                f32_matvec(black_box(&f), inp, out, black_box(&x), &mut y);
                black_box(y.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; out],
            |mut y| {
                for j in 0..out {
                    let row = &f[j * inp..(j + 1) * inp];
                    let mut acc = 0.0f32;
                    for i in 0..inp {
                        acc += row[i] * x[i];
                    }
                    y[j] = acc;
                }
                black_box(y.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let rows = 16;
    let cols = D;
    let x = make(rows * cols);
    let mut group = c.benchmark_group("transpose_16x512");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("simd", |be| {
        be.iter_batched(
            || vec![0.0f32; rows * cols],
            |mut y| {
                transpose_simd(black_box(&x), rows, cols, &mut y);
                black_box(y.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("scalar", |be| {
        be.iter_batched(
            || vec![0.0f32; rows * cols],
            |mut y| {
                for c in 0..cols {
                    for r in 0..rows {
                        y[c * rows + r] = x[r * cols + c];
                    }
                }
                black_box(y.iter().copied().sum::<f32>())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_ln,
    bench_silu,
    bench_glu,
    bench_score,
    bench_softmax,
    bench_dwconv,
    bench_matvec,
    bench_transpose
);
criterion_main!(benches);
