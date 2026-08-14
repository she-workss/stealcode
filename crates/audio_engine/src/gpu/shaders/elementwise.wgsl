// Elementwise kernels, one thread per output element.

struct Params {
    count: u32,
    dim: u32,
    scale: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn sigmoid(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-v));
}

// out[i] = x[i] * sigmoid(x[i])
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    let v = x[i];
    out[i] = v * sigmoid(v);
}

// out[i] = max(0, x[i])
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    out[i] = max(0.0, x[i]);
}

// out[i] = x[i] * sigmoid(x[dim + i])
// GLU over a [t, 2d] input laid out as [gate block | value block]:
// input row is length 2*dim, out row length dim.
@compute @workgroup_size(256)
fn glu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    let row = i / p.dim;
    let off = i % p.dim;
    out[i] = x[row * 2u * p.dim + off]
        * sigmoid(x[row * 2u * p.dim + p.dim + off]);
}

// out[i] = x[i] + scale * y[i]
@compute @workgroup_size(256)
fn add_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    out[i] = x[i] + p.scale * y[i];
}

// out[i] = x[i] + y[i % dim]  (bias add, y is [dim])
@compute @workgroup_size(256)
fn bias_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) {
        return;
    }
    out[i] = x[i] + y[i % p.dim];
}
