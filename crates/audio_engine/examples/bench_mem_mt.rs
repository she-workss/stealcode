//! Multi-threaded raw memory bandwidth probe: 8 threads each read a
//! contiguous 32 MB slice of a shared 256 MB buffer.

use std::time::Instant;

fn main() {
    let total = 256usize * 1024 * 1024 / 4;
    let mut a = vec![0.0f32; total];
    for i in 0..total {
        a[i] = std::hint::black_box(i) as f32;
    }
    let n_threads = 8;
    let per = total / n_threads;
    let t0 = Instant::now();
    let mut s = 0.0f32;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let a = &a;
            handles.push(scope.spawn(move || {
                let mut s = 0.0f32;
                let base = t * per;
                for i in base..base + per {
                    // SAFETY: i < total, a is valid.
                    s += unsafe { std::ptr::read_volatile(a.as_ptr().add(i)) };
                }
                s
            }));
        }
        for h in handles {
            s += h.join().unwrap();
        }
    });
    std::hint::black_box(s);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "8-thread read 256MB: {:.1} GB/s in {:.3}s (sum {s})",
        256.0 / 1024.0 / secs,
        secs
    );
}
