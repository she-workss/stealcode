//! Quick sgemm throughput probe for the encoder shapes (run:
//! cargo run -r -p voice --example sgemm_bench)
use std::time::Instant;

use matrixmultiply::sgemm;

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let t0 = Instant::now();
    for _ in 0..iters {
        voice::nemotron::sgemm_kernel::gemm_into(m, k, n, &a, &b, &mut c);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flop = 2.0 * m as f64 * k as f64 * n as f64 / dt / 1e9;
    println!(
        "{name:32} {m:5}x{k:5}x{n:4}  {:7.3} ms  {:7.1} GF/s",
        dt * 1e3,
        flop
    );
}

fn bench_mm(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c = vec![0.0f32; m * n];
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            sgemm(
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
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flop = 2.0 * m as f64 * k as f64 * n as f64 / dt / 1e9;
    println!(
        "{name:32} {m:5}x{k:5}x{n:4}  {:7.3} ms  {:7.1} GF/s",
        dt * 1e3,
        flop
    );
}

fn main() {
    println!(
        "threads: {:?}",
        std::thread::available_parallelism().map(|n| n.get())
    );
    bench("ff1 lin1 real 0.6b", 4096, 1024, 32, 200);
    bench("ff1 lin2 real", 1024, 4096, 32, 200);
    bench("qkv/attn_out", 768, 768, 32, 200);
    bench("conv pw1", 1536, 768, 34, 200);
    bench("ff1 lin1 n=81", 4096, 1024, 81, 50);
    bench("ff1 lin2 n=81", 1024, 4096, 81, 50);
    bench("ff1 lin1 n=26", 4096, 1024, 26, 50);
    bench("big ref", 4096, 4096, 4096, 10);
    bench_mm("mm ff1 lin1 n=81", 4096, 1024, 81, 50);
    bench_mm("mm ff1 lin2 n=81", 1024, 4096, 81, 50);
    bench_mm("mm ff1 lin1 n=26", 4096, 1024, 26, 50);
    bench_mm("mm ff1 lin1 n=32", 4096, 1024, 32, 200);
    bench_mm("mm ff1 lin2 n=32", 1024, 4096, 32, 200);
    bench_mm("mm qkv n=32", 768, 768, 32, 200);
    bench_mm("mm conv pw1 n=34", 1536, 768, 34, 200);
    bench_mm("mm ff1 lin1 n=19", 4096, 1024, 19, 50);

    // correctness: compare our kernel vs matrixmultiply
    use matrixmultiply::sgemm as mm_sgemm;
    for (m, k, n) in [
        (768usize, 768usize, 32usize),
        (4096, 1024, 32),
        (768, 768, 26),
        (4096, 1024, 81),
        (768, 768, 7),
    ] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7) % 1000) as f32 / 977.0 - 0.4)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13) % 1000) as f32 / 991.0 - 0.3)
            .collect();
        let mut c1 = vec![0.0f32; m * n];
        unsafe {
            mm_sgemm(
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
                c1.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        let mut c2 = vec![0.0f32; m * n];
        voice::nemotron::sgemm_kernel::gemm_into(m, k, n, &a, &b, &mut c2);
        let maxdiff = c1
            .iter()
            .zip(&c2)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let ok = maxdiff < 1e-2 * (m * k) as f32 / 1e3 + 1e-3;
        println!(
            "check {m}x{k}x{n}: maxdiff {maxdiff:.6} {}",
            if ok { "OK" } else { "MISMATCH" }
        );
    }
}
