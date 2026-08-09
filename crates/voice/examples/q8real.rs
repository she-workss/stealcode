//! Reproduce the pre_encode.out linear on real model data, comparing
//! the AVX2 int8 path vs the scalar path.
//! Usage: cargo run -r -p voice --example q8real -- <q8bytes.bin> <xt.bin>
//! <out.bin> <padded_row> <row_len> <n>

use std::path::Path;

use anyhow::Result;
use voice::nemotron::sgemm_kernel::{q8_gemm, q8_gemm_scalar};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let wfile = args.next().unwrap();
    let xfile = args.next().unwrap();
    let out = args.next().unwrap();
    let padded_row: usize = args.next().unwrap().parse()?;
    let row_len: usize = args.next().unwrap().parse()?;
    let n: usize = args.next().unwrap().parse()?;

    let w = std::fs::read(&wfile)?;
    let rows = w.len() / padded_row;
    let x: Vec<f32> = std::fs::read(&xfile)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!(
        "q8mat {rows}x{row_len}, padded_row {padded_row}, x {}x{n}",
        row_len
    );

    let bb = 34usize;
    let qoff = 2usize;
    let mut y_avx = vec![0.0f32; rows * n];
    q8_gemm(rows, row_len, n, &w, padded_row, bb, qoff, &x, &mut y_avx);
    let mut y_scalar = vec![0.0f32; rows * n];
    q8_gemm_scalar(
        rows,
        row_len,
        n,
        &w,
        padded_row,
        bb,
        qoff,
        &x,
        &mut y_scalar,
    );

    let mut maxdiff = 0.0f32;
    for i in 0..y_avx.len() {
        maxdiff = maxdiff.max((y_avx[i] - y_scalar[i]).abs());
    }
    eprintln!("avx vs scalar maxdiff = {maxdiff:.6}");
    eprintln!("avx row0[:4] = {:?}", &y_avx[..4]);
    eprintln!("sca row0[:4] = {:?}", &y_scalar[..4]);

    std::fs::write(&out, {
        let mut v = Vec::with_capacity(y_avx.len() * 4);
        for val in &y_avx {
            v.extend_from_slice(&val.to_le_bytes());
        }
        v
    })?;
    eprintln!("wrote {out}");
    Ok(())
}
