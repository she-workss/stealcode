// Banded attention: scores + softmax + weighted V-sum with relative
// position bias. One workgroup (128 = head_dim lanes) per query frame.
//
//   score[qq, kk, h] = scale * sum_i (q[qq,i] + q_bias[i]) * k[kk,i]
//                              + (q[qq,i] + v_bias[i]) * pos[kk + t - qq - 1, i]
//   out[qq, i] = softmax_kk(score) dot v[kk, i]

struct Params {
    t: u32,
    d: u32,
    n_heads: u32,
    head_dim: u32,
    scale: f32,
    _pad: f32,
    chunk_size: u32,
    left_chunks: u32,
}

const MAX_T: u32 = 512u;

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> q_bias: array<f32>;
@group(0) @binding(5) var<storage, read> v_bias: array<f32>;
@group(0) @binding(6) var<storage, read> pos: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> sdot: array<f32, 128>;
var<workgroup> scores: array<f32, MAX_T>;
var<workgroup> sval: array<f32, 1>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qq = wgid.x;
    if (qq >= p.t) {
        return;
    }
    let t = p.t;
    let d = p.d;
    let hd = p.head_dim;
    let n_h = p.n_heads;
    let lane = lid.x;
    if (lane >= hd) {
        return;
    }
    let q_chunk = qq / p.chunk_size;
    let k_min = (max(q_chunk, p.left_chunks) - p.left_chunks) * p.chunk_size;
    let k_max = min((q_chunk + 1u) * p.chunk_size, t);
    for (var h = 0u; h < n_h; h = h + 1u) {
        let hoff = h * hd;
        let qidx = qq * d + hoff + lane;
        let qui = q[qidx] + q_bias[hoff + lane];
        let qvi = q[qidx] + v_bias[hoff + lane];
        for (var kk = k_min; kk < k_max; kk = kk + 1u) {
            sdot[lane] =
                qui * k[kk * d + hoff + lane]
                + qvi * pos[(kk + t - qq - 1u) * d + hoff + lane];
            workgroupBarrier();
            var stride = 64u;
            while (stride > 0u) {
                if (lane < stride) {
                    sdot[lane] += sdot[lane + stride];
                }
                workgroupBarrier();
                stride = stride >> 1u;
            }
            if (lane == 0u) {
                scores[kk] = sdot[0u] * p.scale;
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            var maxv = -1e30f;
            for (var kk = k_min; kk < k_max; kk = kk + 1u) {
                maxv = max(maxv, scores[kk]);
            }
            var sum = 0.0f;
            for (var kk = k_min; kk < k_max; kk = kk + 1u) {
                let e = exp(scores[kk] - maxv);
                scores[kk] = e;
                sum += e;
            }
            sval[0u] = 1.0 / sum;
        }
        workgroupBarrier();
        let inv = sval[0u];
        var acc = 0.0f;
        for (var kk = k_min; kk < k_max; kk = kk + 1u) {
            acc += scores[kk] * inv * v[kk * d + hoff + lane];
        }
        out[qq * d + hoff + lane] = acc;
        workgroupBarrier();
    }
}
