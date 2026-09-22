//! Braid: three strands plait around the sphere - the "weaving" state.

use std::f32::consts::PI;

use super::{
    core::{Dot, Frame, frac, make_proj, radius_scale, with_fib_dirs},
    profiles::{MAX_GHOST_N, MAX_STRAND_N, ModeOpts, count_usize},
};

pub(super) fn draw_braid_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = (size / 2.0) * 0.76;
    let pt = make_proj(t * 0.4, 0.3, cx, cy, 1.0);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));

    let dots = &mut out.dots;
    let ghost_n = count_usize(o.ghost_n, 150.0, 0, MAX_GHOST_N as usize);
    let strand_n = count_usize(o.strand_n, 52.0, 0, MAX_STRAND_N as usize);
    let inv_r = 1.0 / r;
    dots.reserve(ghost_n.saturating_add(strand_n.saturating_mul(3)));

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

    let turns = o.turns.unwrap_or(3.0);
    let r_base = o.r_base.unwrap_or(1.2);
    let r_depth = o.r_depth.unwrap_or(1.8);
    let inv_strand = 1.0 / strand_n as f32;
    for s in 0..3 {
        let phase = (s as f32 / 3.0) * 2.0 * PI;
        for i in 0..strand_n {
            let u = frac(t.mul_add(0.045, i as f32 * inv_strand))
                .mul_add(2.0, -1.0)
                * 0.96;
            let surf = u.mul_add(-u, 1.0).max(0.0).sqrt();
            let end_fade = ((0.96 - u.abs()) / 0.12).clamp(0.0, 1.0);
            let a = (u * PI).mul_add(turns, phase);
            let weave = 0.075f32.mul_add(
                t.mul_add(0.8, phase.mul_add(2.0, u * PI * turns * 2.0))
                    .sin(),
                1.0,
            );
            let rr = surf * r * weave;
            let (px, py, zr) =
                pt.project(a.cos() * rr, u * r * weave, a.sin() * rr);
            let depth = f32::mul_add(zr, inv_r, 1.0) / 2.0;
            dots.push(
                Dot::new(
                    px,
                    py,
                    zr,
                    r_depth.mul_add(depth, r_base) * rs,
                    0.45f32.mul_add(-depth, 0.55),
                )
                .with_a(end_fade * 0.55f32.mul_add(depth, 0.45)),
            );
        }
    }
}
