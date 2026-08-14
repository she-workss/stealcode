//! Shared math helpers: matrix transposes and byte conversion.

/// Transpose [rows, cols] time-major -> [cols, rows] row-major.
pub(crate) fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            out.push(x[r * cols + c]);
        }
    }
    out
}

/// Like [`transpose`] but reusing a scratch buffer.
pub(crate) fn transpose_into<'a>(
    x: &[f32],
    rows: usize,
    cols: usize,
    trans: &'a mut Vec<f32>,
) -> &'a mut Vec<f32> {
    trans.clear();
    trans.reserve(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            trans.push(x[r * cols + c]);
        }
    }
    trans
}

/// Little-endian byte dump for scratch buffers.
pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
