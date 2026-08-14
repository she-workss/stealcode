// Packed int8 GEMM: y[t, n] = x[t, k] @ W_q8[n, k]^T (+ optional bias).
//
// Weights are stored per output row as `kb = k.div_ceil(32)` blocks:
//   q: array<u32> - 4 int8 weights packed per u32 (i4 of block b at
//      q[((n*kb) + b)*8 + i], i in 0..8)
//   s: array<f32> - block scales, one per (n, b)
// Activations x are [t, k] f32 row-major (no transpose needed on GPU).
// Accumulation is f32; scale applied per 32-block.

struct Params {
    t: u32,     // rows of A
    k: u32,     // cols of A
    n: u32,     // cols of C
    kb: u32,    // weight blocks per output row
    has_bias: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> b: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = gid.x;
    let ni = gid.y;
    if (ti >= p.t || ni >= p.n) {
        return;
    }
    var acc = 0.0f;
    let wrow = ni * p.kb;
    let xrow = ti * p.k;
    for (var kb = 0u; kb < p.kb; kb = kb + 1u) {
        let qbase = wrow * 8u + kb * 8u;
        let xbase = xrow + kb * 32u;
        var bdot = 0.0f;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let w4 = vec4<f32>(unpack4xI8(q[qbase + i]));
            let o = xbase + i * 4u;
            let a4 = vec4<f32>(x[o], x[o + 1u], x[o + 2u], x[o + 3u]);
            bdot += dot(w4, a4);
        }
        acc += s[wrow + kb] * bdot;
    }
    if (p.has_bias != 0u) {
        acc += b[ni];
    }
    out[ti * p.n + ni] = acc;
}
