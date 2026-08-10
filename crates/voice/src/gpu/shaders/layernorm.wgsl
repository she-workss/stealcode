// LayerNorm with affine transform, one workgroup thread per row:
//   out[t, d] = (x[t, d] - mean_t) * rstd_t * gamma[d] + beta[d]

struct Params {
    t: u32,
    d: u32,
    eps: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.t) {
        return;
    }
    let base = t * p.d;
    var sum = 0.0f;
    var sumsq = 0.0f;
    for (var i = 0u; i < p.d; i = i + 1u) {
        let v = x[base + i];
        sum += v;
        sumsq += v * v;
    }
    let mean = sum / f32(p.d);
    let variance = sumsq / f32(p.d) - mean * mean;
    let rstd = 1.0 / sqrt(variance + p.eps);
    for (var i = 0u; i < p.d; i = i + 1u) {
        out[base + i] = (x[base + i] - mean) * rstd * gamma[i] + beta[i];
    }
}
