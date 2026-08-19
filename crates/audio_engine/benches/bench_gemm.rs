//! Criterion benchmarks of the Q8 and f32 GEMM kernels for the
//! streaming encoder shapes (n = 16 = the encoder batch).
//!
//!   cargo bench -p audio_engine --bench bench_gemm
//!
//! Every shape is measured in two regimes:
//!   - `hot`: weights stay in cache (compute ceiling);
//!   - `cold`: a 32 MB flush before every call (the streaming regime, where
//!     each weight matrix is read from DRAM once per batch).

use std::hint::black_box;

use audio_engine::sgemm_kernel::{gemm_into, q8_gemm};
use criterion::{
    BatchSize, Criterion, Throughput, criterion_group, criterion_main,
};

const BLOCK_BYTES: usize = 34;
const QOFF: usize = 2;

struct Setup {
    w: Vec<u8>,
    padded_row: usize,
    x: Vec<f32>,
}

fn setup(m: usize, k: usize, n: usize) -> Setup {
    let nblocks = k.div_ceil(32);
    let padded_row = nblocks * BLOCK_BYTES;
    let mut w = vec![0u8; m * padded_row];
    for (i, b) in w.iter_mut().enumerate() {
        *b = ((i * 7 + 13) % 251) as u8;
    }
    let mut x = vec![0.0f32; k * n];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i * 3) % 101) as f32 / 17.0 - 1.0;
    }
    Setup { w, padded_row, x }
}

fn flush(flush: &mut [u8]) {
    let mut s = 0u64;
    for (i, b) in flush.iter_mut().enumerate() {
        s ^= (*b as u64).wrapping_mul(i as u64);
    }
    black_box(s);
}

fn bench_shape(c: &mut Criterion, name: &str, m: usize, k: usize, n: usize) {
    let s = setup(m, k, n);
    let bytes = m as u64 * s.padded_row as u64;

    let mut hot = c.benchmark_group(format!("{name}/hot"));
    hot.throughput(Throughput::Bytes(bytes));
    hot.bench_function("portable", |b| {
        let mut y = vec![0.0f32; m * n];
        b.iter(|| {
            black_box(q8_gemm(
                m,
                k,
                n,
                &s.w,
                s.padded_row,
                BLOCK_BYTES,
                QOFF,
                &s.x,
                &mut y,
            ))
        });
    });
    hot.finish();

    let mut flush_buf = vec![0u8; 32 * 1024 * 1024];
    let mut cold = c.benchmark_group(format!("{name}/cold"));
    cold.throughput(Throughput::Bytes(bytes));
    // The 32 MB flush runs as `iter_batched` setup, i.e. outside the
    // timed routine, so the measurement is the GEMM alone on
    // DRAM-cold weights (the streaming regime).
    cold.bench_function("portable", |b| {
        let mut y = vec![0.0f32; m * n];
        b.iter_batched(
            || flush(&mut flush_buf),
            |_| {
                black_box(q8_gemm(
                    m,
                    k,
                    n,
                    &s.w,
                    s.padded_row,
                    BLOCK_BYTES,
                    QOFF,
                    &s.x,
                    &mut y,
                ))
            },
            BatchSize::SmallInput,
        );
    });
    cold.finish();
}

fn bench_sgemm_shape(
    c: &mut Criterion,
    name: &str,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut a = vec![0.0f32; m * k];
    let mut bm = vec![0.0f32; k * n];
    for (i, v) in a.iter_mut().enumerate() {
        *v = ((i * 3 + 1) % 97) as f32 / 13.0 - 1.0;
    }
    for (i, v) in bm.iter_mut().enumerate() {
        *v = ((i * 5 + 2) % 89) as f32 / 11.0 - 1.0;
    }
    let bytes = (m as u64 * k as u64 + k as u64 * n as u64) * 4;

    let mut hot = c.benchmark_group(format!("{name}/hot"));
    hot.throughput(Throughput::Bytes(bytes));
    hot.bench_function("portable", |b| {
        let mut y = vec![0.0f32; m * n];
        b.iter(|| {
            black_box(gemm_into(m, k, n, &a, &bm, &mut y));
        });
    });
    hot.finish();

    let mut flush_buf = vec![0u8; 32 * 1024 * 1024];
    let mut cold = c.benchmark_group(format!("{name}/cold"));
    cold.throughput(Throughput::Bytes(bytes));
    cold.bench_function("portable", |b| {
        let mut y = vec![0.0f32; m * n];
        b.iter_batched(
            || flush(&mut flush_buf),
            |_| {
                black_box(gemm_into(m, k, n, &a, &bm, &mut y));
            },
            BatchSize::SmallInput,
        );
    });
    cold.finish();
}

fn gemm_benches(c: &mut Criterion) {
    bench_shape(c, "ff1_lin1", 2048, 512, 16);
    bench_shape(c, "ff1_lin2", 512, 2048, 16);
    bench_shape(c, "qkv", 512, 512, 16);
    bench_shape(c, "scale2048", 2048, 2048, 16);
    bench_sgemm_shape(c, "sff1_lin1", 2048, 512, 16);
    bench_sgemm_shape(c, "sff1_lin2", 512, 2048, 16);
    bench_sgemm_shape(c, "sqkv", 512, 512, 16);
    bench_sgemm_shape(c, "sscale2048", 2048, 2048, 16);
    bench_sgemm_shape(c, "sjoint", 640, 1024, 16);
    bench_sgemm_shape(c, "swide", 2048, 512, 128);
}

criterion_group! {
    name = gemm;
    config = Criterion::default().sample_size(30);
    targets = gemm_benches
}
criterion_main!(gemm);