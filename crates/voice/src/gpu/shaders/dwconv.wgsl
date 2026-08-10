// Causal depthwise 1-D convolution with left padding only:
//   out[tt, c] = sum_k x[tt - pad_left + k, c] * w[c, k]  over valid frames.

struct Params {
    t: u32,
    d: u32,
    kh: u32,
    pad_left: u32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.t * p.d) {
        return;
    }
    let tt = idx / p.d;
    let dd = idx % p.d;
    let base = i32(tt) - i32(p.pad_left);
    var acc = 0.0f;
    for (var k = 0u; k < p.kh; k = k + 1u) {
        let ti = base + i32(k);
        if (ti >= 0 && ti < i32(p.t)) {
            let off = u32(ti) * p.d + dd;
            acc += x[off] * w[dd * p.kh + k];
        }
    }
    out[idx] = acc;
}
