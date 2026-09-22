//! Original low-density orb concepts: focus, gyroscope, and memory echoes.
//!
//! These modes deliberately reuse the engine's existing count/radius knobs and
//! emit dots only, keeping them on the cheapest GPUI paint path.

use std::f32::consts::PI;

use super::{
    core::{Dot, Frame, frac, make_proj, radius_scale, with_unit_circle},
    profiles::{MAX_LANES, MAX_PARTICLES, MAX_SEGS, ModeOpts, count_usize},
};

/// Iris-like streams converge on a small, steady focal core.
pub(super) fn draw_focus_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let center = size / 2.0;
    let extent = size * 0.38 * o.spread.unwrap_or(1.0);
    let lanes = count_usize(o.lanes, 6.0, 1, MAX_LANES as usize);
    let segs = count_usize(o.segs, 12.0, 1, MAX_SEGS as usize);
    let core_n = count_usize(o.particles, 5.0, 1, MAX_PARTICLES as usize);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let r_base = o.r_base.unwrap_or(0.9);
    let r_depth = o.r_depth.unwrap_or(1.25);
    let total = lanes.saturating_mul(segs).saturating_add(core_n);
    out.dots.reserve(total);

    // Close and reopen a six-blade aperture without collapsing the samples
    // into a star. Every blade keeps a continuous curved silhouette; only its
    // inner edge moves, so the concept remains readable in a frozen frame.
    let openness = 0.5f32.mul_add((t * 0.9).sin(), 0.5);
    let inner = 0.2f32.mul_add(openness, 0.12);
    let rotation = t * 0.11;
    let inv_segs = 1.0 / segs as f32;
    with_unit_circle(lanes, |directions| {
        for (lane, direction) in directions.iter().enumerate() {
            for i in 0..segs {
                let u = (i as f32 + 0.5) * inv_segs;
                let radial = (1.0 - inner).mul_add(u, inner);
                let curl =
                    0.24f32.mul_add(-openness, 0.88) * (1.0 - u).powf(1.25);
                let twist = rotation + curl;
                let (st, ct) = twist.sin_cos();
                let dx = f32::mul_add(direction[1], -st, direction[0] * ct);
                let dy = f32::mul_add(direction[1], ct, direction[0] * st);
                let inner_weight = 1.0 - u;
                let shimmer = 0.05f32
                    .mul_add((lane as f32).mul_add(0.8, t * 1.4).sin(), 0.96);
                out.dots.push(
                    Dot::new(
                        (dx * extent).mul_add(radial, center),
                        (dy * extent).mul_add(radial, center),
                        0.0,
                        r_depth.mul_add(inner_weight, r_base) * shimmer * rs,
                        0.52f32.mul_add(u, 0.12),
                    )
                    .with_a(0.52f32.mul_add((PI * u).sin(), 0.48)),
                );
            }
        }
    });

    // A tiny breathing nucleus makes the destination legible at inline size.
    let core_radius = size * 0.004f32.mul_add((t * 1.5).sin(), 0.035);
    with_unit_circle(core_n, |circle| {
        for p in circle {
            out.dots.push(Dot::new(
                f32::mul_add(p[0], core_radius, center),
                f32::mul_add(p[1], core_radius, center),
                0.0,
                r_depth.mul_add(1.15, r_base) * rs,
                0.08,
            ));
        }
    });
}

/// Three inference loops carry thoughts in alternating directions.
pub(super) fn draw_gyroscope_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let center = size / 2.0;
    let radius = size * 0.385 * o.spread.unwrap_or(1.0);
    let rings = count_usize(o.lanes, 3.0, 1, MAX_LANES as usize).min(6);
    let segs = count_usize(o.segs, 24.0, 3, MAX_SEGS as usize);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let r_base = o.r_base.unwrap_or(0.8);
    let r_depth = o.r_depth.unwrap_or(1.5);
    let proj = make_proj(t * 0.055, 0.18, center, center, radius);
    out.dots
        .reserve(rings.saturating_mul(segs.saturating_add(2)));

    with_unit_circle(segs, |circle| {
        for ring in 0..rings {
            let spread = if rings > 1 {
                ring as f32 / (rings - 1) as f32 - 0.5
            } else {
                0.0
            };
            let tilt_x = 0.12f32
                .mul_add(t.mul_add(0.16, ring as f32).sin(), spread * 1.55);
            let tilt_y = ring as f32 * PI / rings as f32 + 0.32;
            let (sx, cx) = tilt_x.sin_cos();
            let (sy, cy) = tilt_y.sin_cos();
            let direction = if ring % 2 == 0 { 1.0 } else { -1.0 };
            let orient = |ca: f32, sa: f32| {
                let y1 = sa * cx;
                let z1 = sa * sx;
                (z1.mul_add(sy, ca * cy), y1, z1.mul_add(cy, -ca * sy))
            };

            for p in circle {
                let (x, y, z) = orient(p[0], p[1]);
                let (px, py, depth_z) = proj.project(x, y, z);
                let depth = f32::midpoint(depth_z, 1.0);
                out.dots.push(
                    Dot::new(
                        px,
                        py,
                        depth_z,
                        (r_depth * 0.5).mul_add(depth, r_base * 0.9) * rs,
                        0.34f32.mul_add(-depth, 0.62),
                    )
                    .with_a(0.4f32.mul_add(depth, 0.34)),
                );
            }

            // A leading thought and a softer echo make direction and velocity
            // legible without turning the complete track into visual noise.
            let travel = (t * (ring as f32).mul_add(0.08, 0.72))
                .mul_add(direction, ring as f32 * 2.0 * PI / rings as f32);
            for (trail, offset) in
                [0.0, -0.22 * direction].into_iter().enumerate()
            {
                let (sa, ca) = (travel + offset).sin_cos();
                let (x, y, z) = orient(ca, sa);
                let (px, py, depth_z) = proj.project(x, y, z);
                let depth = f32::midpoint(depth_z, 1.0);
                let strength = if trail == 0 { 1.0 } else { 0.48 };
                out.dots.push(
                    Dot::new(
                        px,
                        py,
                        depth_z + 0.002,
                        1.65f32
                            .mul_add(strength, r_depth.mul_add(depth, r_base))
                            * rs,
                        0.2f32
                            .mul_add(-strength, 0.42f32.mul_add(-depth, 0.44)),
                    )
                    .with_a(0.45f32.mul_add(depth, 0.55) * strength),
                );
            }
        }
    });
}

/// Concentric echoes emerge from a stable core and dissolve at the boundary.
pub(super) fn draw_echo_into(size: f32, t: f32, o: &ModeOpts, out: &mut Frame) {
    let center = size / 2.0;
    let extent = size * 0.4 * o.spread.unwrap_or(1.0);
    let rings = count_usize(o.lanes, 4.0, 1, MAX_LANES as usize);
    let segs = count_usize(o.segs, 18.0, 3, MAX_SEGS as usize);
    let core_n = count_usize(o.particles, 3.0, 1, MAX_PARTICLES as usize);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let r_base = o.r_base.unwrap_or(0.85);
    let r_depth = o.r_depth.unwrap_or(1.05);
    out.dots
        .reserve(rings.saturating_mul(segs).saturating_add(core_n));

    with_unit_circle(segs, |circle| {
        for ring in 0..rings {
            let phase = frac(t.mul_add(0.14, ring as f32 / rings as f32));
            let radius = extent * 0.87f32.mul_add(phase, 0.13);
            let life = (PI * phase).sin().max(0.0);
            let turn = t.mul_add(-0.1, ring as f32 * 0.37);
            let (st, ct) = turn.sin_cos();
            for p in circle {
                let x = f32::mul_add(p[1], -st, p[0] * ct);
                let y = f32::mul_add(p[1], ct, p[0] * st);
                out.dots.push(
                    Dot::new(
                        f32::mul_add(x, radius, center),
                        f32::mul_add(y, radius, center),
                        0.0,
                        r_depth.mul_add(1.0 - phase, r_base) * rs,
                        0.52f32.mul_add(phase, 0.18),
                    )
                    .with_a(life.powf(0.58)),
                );
            }
        }
    });

    let core_radius = size * 0.025;
    with_unit_circle(core_n, |circle| {
        for p in circle {
            out.dots.push(Dot::new(
                f32::mul_add(p[0], core_radius, center),
                f32::mul_add(p[1], core_radius, center),
                0.0,
                r_depth.mul_add(1.4, r_base) * rs,
                0.06,
            ));
        }
    });
}
