//! Orbits: particles on tilted orbits - the "working" state.

use std::f32::consts::PI;

use super::{
    core::{Dot, Frame, hash_d, make_proj, radius_scale, with_unit_circle},
    profiles::{
        MAX_GHOST_N, MAX_ORBIT_N, MAX_PARTICLES, ModeOpts, count_usize,
    },
};

pub(super) fn draw_orbits_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = (size / 2.0) * 0.82;
    let pt = make_proj(t * 0.12, 0.3, cx, cy, 1.0);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));

    let orbit_n = count_usize(o.orbit_n, 12.0, 0, MAX_ORBIT_N as usize);
    let ghost_n = count_usize(o.ghost_n, 40.0, 0, MAX_GHOST_N as usize);
    let particles = count_usize(o.particles, 3.0, 0, MAX_PARTICLES as usize);

    let dots = &mut out.dots;
    let per_orbit = ghost_n.saturating_add(particles);
    dots.reserve(orbit_n.saturating_mul(per_orbit));
    let ghost_r = o.ghost_r.unwrap_or(0.9) * rs;
    let ghost_a = o.ghost_a.unwrap_or(0.5);
    let part_r = o.part_r.unwrap_or(1.2);
    let part_r_depth = o.part_r_depth.unwrap_or(1.6);

    for orb in 0..orbit_n {
        let orb_f = orb as f32;
        let h1 = hash_d(orb_f, 1.7);
        let h2 = hash_d(orb_f, 5.2);
        let h3 = hash_d(orb_f, 8.9);
        let ro = r * 0.52f32.mul_add(h1, 0.45);
        let th = h1 * 2.0 * PI;
        let phi = 2.0f32.mul_add(h2, -1.0).acos();
        // Build a stable plane basis by crossing the normal with whichever
        // world axis is least parallel to it.
        let nx = phi.sin() * th.cos();
        let ny = phi.cos();
        let nz = phi.sin() * th.sin();
        let (mut ux, mut uy, mut uz) = if nz.abs() < 0.9 {
            (-ny, nx, 0.0)
        } else {
            (0.0, -nz, ny)
        };
        let ul = f32::mul_add(uz, uz, f32::mul_add(uy, uy, ux * ux))
            .sqrt()
            .max(1e-6);
        ux /= ul;
        uy /= ul;
        uz /= ul;
        let vx = nz.mul_add(-uy, ny * uz);
        let vy = nx.mul_add(-uz, nz * ux);
        let vz = ny.mul_add(-ux, nx * uy);
        let speed =
            0.55f32.mul_add(h3, 0.25) * if h3 > 0.5 { 1.0 } else { -1.0 };
        let inv_ro = 1.0 / ro;

        // Ghost topology is invariant; reuse its unit-circle samples instead
        // of evaluating 2 × orbit_n × ghost_n trig functions every frame.
        with_unit_circle(ghost_n, |circle| {
            for p in circle {
                let (ca, sa) = (*p).into();
                let (px, py, z) = pt.project(
                    vx.mul_add(sa, ux * ca) * ro,
                    vy.mul_add(sa, uy * ca) * ro,
                    vz.mul_add(sa, uz * ca) * ro,
                );
                let depth = f32::mul_add(z, inv_ro, 1.0) / 2.0;
                dots.push(
                    Dot::new(px, py, z, ghost_r, 0.72)
                        .with_a(ghost_a * 0.6f32.mul_add(depth, 0.4)),
                );
            }
        });
        // the particles doing the work
        for m in 0..particles {
            let a = h2.mul_add(
                6.0,
                ((m as f32 / particles as f32) * 2.0).mul_add(PI, t * speed),
            );
            let (ca, sa) = (a.cos(), a.sin());
            let (px, py, z) = pt.project(
                vx.mul_add(sa, ux * ca) * ro,
                vy.mul_add(sa, uy * ca) * ro,
                vz.mul_add(sa, uz * ca) * ro,
            );
            let depth = f32::mul_add(z, inv_ro, 1.0) / 2.0;
            dots.push(Dot::new(
                px,
                py,
                z,
                part_r_depth.mul_add(depth, part_r) * rs,
                0.22f32.mul_add(-depth, 0.3),
            ));
        }
    }
}
