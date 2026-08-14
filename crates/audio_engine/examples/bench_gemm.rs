//! Micro-benchmark of the Q8 GEMM kernel for thin batches.
//!
//!   cargo run -r -p audio_engine --example bench_gemm
//!
//! Times q8_gemm for the shapes the streaming encoder actually uses
//! (n = 1 vs n = 14). For n <= 4 the row-sequential thin kernel runs
//! (contiguous weight streams + rayon row chunks); wider batches use
//! the 8-row AVX2 tile over rayon.

use audio_engine::sgemm_kernel::q8_gemm;
use std::time::Instant;

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let nblocks = k.div_ceil(32);
    let block_bytes = 34;
    let padded_row = nblocks * block_bytes;
    let mut w = vec![0u8; m * padded_row];
    for (i, b) in w.iter_mut().enumerate() {
        *b = ((i * 7 + 13) % 251) as u8;
    }
    let mut x = vec![0.0f32; k * n];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i * 3) % 101) as f32 / 17.0 - 1.0;
    }
    let mut y = vec![0.0f32; m * n];
    // warmup
    for _ in 0..50 {
        q8_gemm(m, k, n, &w, padded_row, block_bytes, 2, &x, &mut y);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        q8_gemm(m, k, n, &w, padded_row, block_bytes, 2, &x, &mut y);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!(
        "{name:>22} [{m:>5}x{k:>5}x{n:>2}] {per:8.1} µs/call"
    );
}

fn main() {
    bench("ff1_lin1 n=1", 2048, 512, 1, 2000);
    bench("ff1_lin1 n=14", 2048, 512, 14, 500);
    bench("ff1_lin2 n=1", 512, 2048, 1, 2000);
    bench("ff1_lin2 n=14", 512, 2048, 14, 500);
    bench("qkv n=1", 512, 512, 1, 4000);
    bench("qkv n=14", 512, 512, 14, 1000);
    bench("pw1 n=1", 4096, 512, 1, 2000);
    bench("pw1 n=14", 4096, 512, 14, 500);
    // stream-length scaling (n = 16, the encoder's batch): does the
    // aggregate rate rise with bigger per-thread contiguous streams?
    bench("scale m=512  n=16", 512, 2048, 16, 400);
    bench("scale m=2048 n=16", 2048, 2048, 16, 200);
}
