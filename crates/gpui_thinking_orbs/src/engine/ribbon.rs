//! Ribbon / ring: undulating sash or face-on breathing ring.

use std::f32::consts::PI;

use super::{
    core::{Dot, Frame, make_proj, radius_scale, with_fib_dirs},
    profiles::{MAX_GHOST_N, MAX_LANES, MAX_SEGS, ModeOpts, count_usize},
};

pub(super) fn draw_ribbon_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = (size / 2.0) * 0.78;
    let spin = o.spin.unwrap_or(1.0);
    let cam_tilt = 0.3;
    let pt = make_proj(t * 0.1 * spin, cam_tilt, cx, cy, 1.0);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let face_on = o.face_on.unwrap_or(0.0) != 0.0;

    let dots = &mut out.dots;
    let ghost_n = count_usize(o.ghost_n, 150.0, 0, MAX_GHOST_N as usize);
    let inv_r = 1.0 / r;
    dots.reserve(ghost_n);
    with_fib_dirs(ghost_n, |directions| {
        for d in directions {
            let (px, py, z) = pt.project(d[0] * r, d[1] * r, d[2] * r);
            let depth = f32::mul_add(z, inv_r, 1.0) / 2.0;
            dots.push(
                Dot::new(px, py, z, 0.8 * rs, 0.78)
                    .with_a(0.22f32.mul_add(depth, 0.1)),
            );
        }
    });

    let ya = t * 0.24 * spin;
    let ta = if face_on {
        -cam_tilt
    } else {
        (0.3 * (t * 0.18).sin()).mul_add(spin, 0.55)
    };
    let ux = ya.cos();
    let uy = 0.0;
    let uz = ya.sin();
    let (sta, cta) = (ta.sin(), ta.cos());
    let vx = -uz * sta;
    let vy = cta;
    let vz = ux * sta;
    // plane normal n = u × v
    let nx = uz.mul_add(-vy, uy * vz);
    let ny = ux.mul_add(-vz, uz * vx);
    let nz = f32::mul_add(uy, -vx, ux * vy);

    let wob_mul = o.wob_mul.unwrap_or(1.0);
    let wob_amp = 0.23 * wob_mul;
    let base_r = if face_on {
        r / 0.85f32.mul_add(wob_amp, 1.0)
    } else {
        r
    };

    let base_lanes = o.lanes.unwrap_or(5.0).clamp(1.0, MAX_LANES);
    let segs = count_usize(o.segs, 88.0, 1, MAX_SEGS as usize);
    let lanes = (base_lanes * o.band_mul.unwrap_or(1.0))
        .round()
        .clamp(1.0, MAX_LANES) as usize;
    let r_base = o.r_base.unwrap_or(1.1);
    let r_depth = o.r_depth.unwrap_or(1.7);
    let seg_step = 2.0 * PI / segs as f32;
    let half = (lanes as f32 - 1.0) / 2.0;
    let inv_half = 1.0 / half.max(1.0);
    dots.reserve(lanes.saturating_mul(segs));

    for w in 0..lanes {
        let centered = w as f32 - half;
        let lane_off = centered * 0.075;
        let edge = centered.abs() * inv_half;
        let edge_r = 0.25f32.mul_add(-edge, 1.0);
        let edge_ink = 0.18 * edge;
        for k in 0..segs {
            let a = k as f32 * seg_step;
            let (ca, sa) = (a.cos(), a.sin());
            let breath = if face_on {
                0.28f32.mul_add((t * 0.48).sin(), 0.72)
            } else {
                1.0
            };
            let wob = 0.07f32.mul_add(
                t.mul_add(1.1, a * 5.0).sin(),
                0.16 * (w as f32).mul_add(0.22, t.mul_add(-1.7, a * 3.0)).sin(),
            ) * wob_mul
                * breath;
            let radial = if face_on { 1.0 + wob } else { 1.0 };
            let off = if face_on { lane_off } else { lane_off + wob };
            let x = f32::mul_add(nx, off, vx.mul_add(sa, ux * ca));
            let y = ny.mul_add(off, f32::mul_add(vy, sa, uy * ca));
            let z = f32::mul_add(nz, off, vz.mul_add(sa, uz * ca));
            let l = f32::mul_add(z, z, y.mul_add(y, x * x)).sqrt();
            let inv_l = if l > 1e-6 { 1.0 / l } else { 0.0 };
            let rr = base_r * radial * inv_l;
            let (px, py, zr) = pt.project(x * rr, y * rr, z * rr);
            let depth = f32::mul_add(zr, inv_r, 1.0) / 2.0;
            dots.push(
                Dot::new(
                    px,
                    py,
                    zr,
                    r_depth.mul_add(depth, r_base) * edge_r * rs,
                    0.44f32.mul_add(-depth, 0.52) + edge_ink,
                )
                .with_a(0.6f32.mul_add(depth, 0.4)),
            );
        }
    }
}
