//! Long-running aggregate DRAM read test (boosts the CPU, settles into
//! the machine's real sustained bandwidth).

use std::time::Instant;

fn main() {
    let n = 256usize * 1024 * 1024 / 4;
    let mut a = vec![0u32; n];
    for i in 0..n {
        a[i] = (i as u32).wrapping_mul(2654435761);
    }
    std::hint::black_box(&a);
    let n_threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let t0 = Instant::now();
    let mut s = 0u64;
    for _ in 0..iters {
        std::thread::scope(|scope| {
            let mut hs = Vec::new();
            for t in 0..n_threads {
                let a = &a;
                hs.push(scope.spawn(move || {
                    let mut s0 = 0u64;
                    let mut s1 = 0u64;
                    let mut s2 = 0u64;
                    let mut s3 = 0u64;
                    let mut s4 = 0u64;
                    let mut s5 = 0u64;
                    let mut s6 = 0u64;
                    let mut s7 = 0u64;
                    let base = t * n / n_threads;
                    let end = base + n / n_threads;
                    let mut i = base;
                    while i + 7 < end {
                        s0 ^= a[i] as u64;
                        s1 ^= a[i + 1] as u64;
                        s2 ^= a[i + 2] as u64;
                        s3 ^= a[i + 3] as u64;
                        s4 ^= a[i + 4] as u64;
                        s5 ^= a[i + 5] as u64;
                        s6 ^= a[i + 6] as u64;
                        s7 ^= a[i + 7] as u64;
                        i += 8;
                    }
                    s0 ^ s1 ^ s2 ^ s3 ^ s4 ^ s5 ^ s6 ^ s7
                }));
            }
            for h in hs {
                s ^= h.join().unwrap();
            }
        });
    }
    std::hint::black_box(s);
    let secs = t0.elapsed().as_secs_f64();
    let gb = iters as f64 * n as f64 * 4.0 / 1e9;
    println!(
        "{n_threads} threads x{iters}: {gb:.1} GB in {secs:.3}s = {:.1} GB/s (sum {s})",
        gb / secs
    );
}
