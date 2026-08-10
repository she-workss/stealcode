// Streaming attention: new query frames against a combined K/V cache.
// One workgroup (128 = head_dim lanes) per new query frame.
//
//   score[qi, kk, h] = scale * sum_i (q[qi,i] + pos_u[i]) * kv[kk,i]
//                              + (q[qi,i] + pos_v[i]) * pos_p[qq - fr + pos_off, i]
//   out[qi, i] = softmax_kk(score) dot vv[kk, i]
//
// where `kv`/`vv` are the band's frames [k_lo, k_hi) (old cached + new)
// and `qq = s + qi` is the absolute frame index.

struct Params {
    c: u32,
    d: u32,
    n_heads: u32,
    head_dim: u32,
    scale: f32,
    s: u32,
    k_lo: u32,
    band: u32,
    chunk: u32,
    left_chunks: u32,
    k_hi: u32,
    pos_off: u32,
}

const MAX_T: u32 = 512u;

@group(0) @binding(0) var<storage, read> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> vv: array<f32>;
@group(0) @binding(4) var<storage, read> pos: array<f32>;
@group(0) @binding(5) var<storage, read> pos_u: array<f32>;
@group(0) @binding(6) var<storage, read> pos_v: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> sdot: array<f32, 128>;
var<workgroup> scores: array<f32, MAX_T>;
var<workgroup> sval: array<f32, 1>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qi = wgid.x;
    if (qi >= p.c) {
        return;
    }
    let qq = p.s + qi;
    let q_chunk = qq / p.chunk;
    let k_min = (max(q_chunk, p.left_chunks) - p.left_chunks) * p.chunk;
    let k_max = min((q_chunk + 1u) * p.chunk, p.k_hi);
    let k0 = k_min - p.k_lo;
    let k1 = k_max - p.k_lo;
    let lane = lid.x;
    if (lane >= p.head_dim) {
        return;
    }
    let d = p.d;
    let hd = p.head_dim;
    for (var h = 0u; h < p.n_heads; h = h + 1u) {
        let hoff = h * hd;
        let qidx = qi * d + hoff + lane;
        let qui = q[qidx] + pos_u[hoff + lane];
        let qvi = q[qidx] + pos_v[hoff + lane];
        for (var kk = k0; kk < k1; kk = kk + 1u) {
            let fr = p.k_lo + kk;
            let pr = i32(qq) - i32(fr) + i32(p.pos_off);
            sdot[lane] =
                qui * kv[kk * d + hoff + lane]
                + qvi * pos[u32(pr) * d + hoff + lane];
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
            for (var kk = k0; kk < k1; kk = kk + 1u) {
                maxv = max(maxv, scores[kk]);
            }
            var sum = 0.0f;
            for (var kk = k0; kk < k1; kk = kk + 1u) {
                let e = exp(scores[kk] - maxv);
                scores[kk] = e;
                sum += e;
            }
            sval[0u] = 1.0 / sum;
        }
        workgroupBarrier();
        let inv = sval[0u];
        var acc = 0.0f;
        for (var kk = k0; kk < k1; kk = kk + 1u) {
            acc += scores[kk] * inv * vv[kk * d + hoff + lane];
        }
        out[qi * d + hoff + lane] = acc;
        workgroupBarrier();
    }
}
