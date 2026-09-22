//! Sphere-lattice modes: globe (searching), rubik (solving), wave (listening).

use std::{cell::RefCell, f32::consts::PI};

use super::{
    core::{
        Dot, Frame, angle_delta, hash_d, make_proj, radius_scale,
        with_unit_circle,
    },
    profiles::{
        MAX_LATTICE_RINGS, MAX_LON_DENSITY, MAX_MOVE_COUNT, ModeOpts,
        count_usize,
    },
};

struct Move {
    axis: u8,
    lo: f32,
    hi: f32,
    ang: f32,
}

thread_local! {
    /// Scratch for Rubik band moves - reused every frame so Solving is
    /// allocation-free in steady state.
    static MOVES: RefCell<Vec<Move>> = const { RefCell::new(Vec::new()) };
    static AMOUNTS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn solve_cycle_into(
    time: f32,
    count: usize,
    slot_dur: f32,
    rest: f32,
    amount: &mut Vec<f32>,
) -> (i32, f32) {
    amount.clear();
    amount.resize(count, 0.0);
    let mut active = -1_i32;
    if count == 0 {
        return (active, 0.0);
    }
    let cyc = (2.0 * count as f32).mul_add(slot_dur, rest);
    let tc = time % cyc;
    if tc < 2.0 * count as f32 * slot_dur {
        // The guard above and this division are two independent float
        // operations, so rounding can let `tc / slot_dur` reach `2 * count`
        // even though `tc < 2 * count * slot_dur` held - e.g. count = 9,
        // slot_dur = 0.42, tc = 7.559999466. Without the clamp `2*count-1-slot`
        // wraps to `usize::MAX` and the loop below indexes out of bounds.
        let slot = ((tc / slot_dur).floor() as usize).min(2 * count - 1);
        let p = (slot as f32).mul_add(-slot_dur, tc) / slot_dur;
        let p = p.clamp(0.0, 1.0);
        let ep = p * p * 2.0f32.mul_add(-p, 3.0);
        let emphasis = (PI * p).sin().max(0.0).sqrt();
        if slot < count {
            for a in amount.iter_mut().take(slot) {
                *a = 1.0;
            }
            amount[slot] = ep;
            active = slot as i32;
        } else {
            let u = 2 * count - 1 - slot;
            for a in amount.iter_mut().take(u) {
                *a = 1.0;
            }
            if u < count {
                amount[u] = 1.0 - ep;
            }
            active = u as i32;
        }
        return (active, emphasis);
    }
    (active, 0.0)
}

fn apply_moves(
    pt3: (f32, f32, f32),
    moves: &[Move],
    amounts: &[f32],
    active: i32,
) -> (f32, f32, f32, bool) {
    let (mut x, mut y, mut z) = pt3;
    let mut in_active = false;
    for (i, mv) in moves.iter().enumerate() {
        if amounts[i] <= 0.0 {
            continue;
        }
        let coord = match mv.axis {
            0 => x,
            1 => y,
            _ => z,
        };
        if coord < mv.lo || coord >= mv.hi {
            continue;
        }
        if i as i32 == active {
            in_active = true;
        }
        let a = mv.ang * amounts[i];
        let ca = a.cos();
        let sa = a.sin();
        match mv.axis {
            0 => {
                let y2 = f32::mul_add(z, -sa, y * ca);
                z = f32::mul_add(z, ca, y * sa);
                y = y2;
            }
            1 => {
                let x2 = f32::mul_add(z, sa, x * ca);
                z = f32::mul_add(z, ca, -x * sa);
                x = x2;
            }
            _ => {
                let x2 = f32::mul_add(y, -sa, x * ca);
                y = f32::mul_add(y, ca, x * sa);
                x = x2;
            }
        }
    }
    (x, y, z, in_active)
}

fn make_moves_into(count: usize, moves: &mut Vec<Move>) {
    moves.clear();
    moves.reserve(count);
    for i in 0..count {
        let i_f = i as f32;
        let axis = (hash_d(i_f, 2.3) * 3.0).floor().min(2.0) as u8;
        let lo =
            0.5f32.mul_add((hash_d(i_f, 5.9) * 4.0).floor().min(3.0), -1.0);
        let dir = if hash_d(i_f, 7.7) < 0.5 { 1.0 } else { -1.0 };
        moves.push(Move {
            axis,
            lo,
            hi: lo + 0.5,
            ang: dir * PI / 2.0,
        });
    }
}

/// Globe: lat/long field, a scan meridian sweeps - searching.
pub(super) fn draw_globe_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let spin = 0.5;
    let cx = size / 2.0;
    let cy = size / 2.0;
    let radius = (size / 2.0) * 0.82;
    let tilt = 0.06f32.mul_add((t * 0.35).sin(), 0.4);
    let pt = make_proj(t * spin, tilt, cx, cy, radius);
    let scan = t * f32::mul_add(1.7 - spin, o.scan_mul.unwrap_or(1.0), spin);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let dim_base = o.dim_base.unwrap_or(1.0);

    let dots = &mut out.dots;
    let lat_rings =
        count_usize(o.lat_rings, 17.0, 1, MAX_LATTICE_RINGS as usize);
    let lon_density = o.lon_density.unwrap_or(44.0).clamp(1.0, MAX_LON_DENSITY);
    let r_base = o.r_base.unwrap_or(0.6);
    let r_depth = o.r_depth.unwrap_or(1.7);
    let r_boost = o.r_boost.unwrap_or(1.0);
    let ink_far = o.ink_far.unwrap_or(0.62);
    let ink_span = o.ink_span.unwrap_or(0.54);
    let inv_lat = 1.0 / lat_rings as f32;

    for li in 0..=lat_rings {
        let lat = (li as f32 * inv_lat).mul_add(PI, -PI / 2.0);
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();
        let lon_count = ((cos_lat.abs() * lon_density).round() as usize).max(1);
        let lon_step = 2.0 * PI / lon_count as f32;
        with_unit_circle(lon_count, |circle| {
            for (lj, p) in circle.iter().enumerate() {
                let lon = lj as f32 * lon_step;
                let (clon, slon) = (*p).into();
                let (px, py, z) =
                    pt.project(cos_lat * clon, sin_lat, cos_lat * slon);
                let depth = f32::midpoint(z, 1.0);
                let d = angle_delta(t.mul_add(spin, lon), scan);
                let boost = (-(d * d) / 0.18).exp() * z.max(0.0);
                dots.push(
                    Dot::new(
                        px,
                        py,
                        z,
                        r_boost.mul_add(boost, r_depth.mul_add(depth, r_base))
                            * rs,
                        ink_span.mul_add(-depth, ink_far),
                    )
                    .with_a((1.0 - dim_base).mul_add(boost.min(1.0), dim_base)),
                );
            }
        });
    }
}

/// Rubik: bands twist in quarter turns, scramble → solve - solving.
pub(super) fn draw_rubik_into(
    size: f32,
    t: f32,
    o: &ModeOpts,
    out: &mut Frame,
) {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = (size / 2.0) * 0.82;
    // Keep the lattice almost still so each quarter-turn reads as the action,
    // rather than getting lost inside a continuously spinning globe.
    let pt = make_proj(
        t * 0.12,
        0.035f32.mul_add((t * 0.45).sin(), 0.35),
        cx,
        cy,
        r,
    );
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));
    let move_count =
        count_usize(o.move_count, 14.0, 0, MAX_MOVE_COUNT as usize);

    MOVES.with(|moves_cell| {
        AMOUNTS.with(|amounts_cell| {
            let mut moves = moves_cell.borrow_mut();
            let mut amounts = amounts_cell.borrow_mut();
            if moves.len() != move_count {
                make_moves_into(move_count, &mut moves);
            }
            let (active, emphasis) =
                solve_cycle_into(t, move_count, 0.58, 1.35, &mut amounts);

            let dots = &mut out.dots;
            let lat_rings =
                count_usize(o.lat_rings, 15.0, 1, MAX_LATTICE_RINGS as usize);
            let lon_density =
                o.lon_density.unwrap_or(40.0).clamp(1.0, MAX_LON_DENSITY);
            let r_base = o.r_base.unwrap_or(0.6);
            let r_depth = o.r_depth.unwrap_or(1.7);
            let r_active = o.r_active.unwrap_or(0.3);
            let ink_far = o.ink_far.unwrap_or(0.62);
            let ink_span = o.ink_span.unwrap_or(0.54);
            let inv_lat = 1.0 / lat_rings as f32;

            for li in 0..=lat_rings {
                let lat = (li as f32 * inv_lat).mul_add(PI, -PI / 2.0);
                let cos_lat = lat.cos();
                let sin_lat = lat.sin();
                let lon_count =
                    ((cos_lat.abs() * lon_density).round() as usize).max(1);
                with_unit_circle(lon_count, |circle| {
                    for p in circle {
                        let (x, y, z, in_active) = apply_moves(
                            (cos_lat * p[0], sin_lat, cos_lat * p[1]),
                            &moves,
                            &amounts,
                            active,
                        );
                        let (px, py, zr) = pt.project(x, y, z);
                        let depth = f32::midpoint(zr, 1.0);
                        let active_mix = if in_active { emphasis } else { 0.0 };
                        dots.push(
                            Dot::new(
                                px,
                                py,
                                zr,
                                (r_active * 2.1).mul_add(
                                    active_mix,
                                    r_depth.mul_add(depth, r_base),
                                ) * rs,
                                0.24f32.mul_add(
                                    -active_mix,
                                    ink_span.mul_add(-depth, ink_far),
                                ),
                            )
                            .with_a(
                                0.28f32.mul_add(
                                    active_mix,
                                    0.14f32.mul_add(depth, 0.58),
                                ),
                            ),
                        );
                    }
                });
            }
        });
    });
}

/// Wave: a waveform rolls through the rings - listening.
pub(super) fn draw_wave_into(size: f32, t: f32, o: &ModeOpts, out: &mut Frame) {
    let cx = size / 2.0;
    let cy = size / 2.0;
    let r = (size / 2.0) * 0.874;
    let pt = make_proj(t * 0.18, 0.38, cx, cy, 1.0);
    let rs = radius_scale(size, o.rs_pow.unwrap_or(0.6));

    let dots = &mut out.dots;
    let rings = count_usize(o.rings, 15.0, 1, MAX_LATTICE_RINGS as usize);
    let lon_density = o.lon_density.unwrap_or(40.0).clamp(1.0, MAX_LON_DENSITY);
    let r_base = o.r_base.unwrap_or(0.6);
    let r_depth = o.r_depth.unwrap_or(1.7);
    let inv_rings = 1.0 / rings as f32;
    let inv_r = 1.0 / r;

    for ri in 0..=rings {
        let lat = (ri as f32 * inv_rings).mul_add(PI, -PI / 2.0);
        let cos_lat = lat.cos();
        let sin_lat = lat.sin();
        let w = 0.38f32.mul_add(
            (ri as f32).mul_add(0.83, t * 1.27).sin(),
            0.62 * (ri as f32).mul_add(-0.52, t * 2.1).sin(),
        );
        let rr = r * 0.105f32.mul_add(w, 0.88);
        let lon_count = ((cos_lat.abs() * lon_density).round() as usize).max(1);
        let crest = w.max(0.0);
        let crest_r = 0.4f32.mul_add(crest, 1.0);
        let crest_ink = 0.1 * crest;
        with_unit_circle(lon_count, |circle| {
            for p in circle {
                let (clon, slon) = (*p).into();
                let (px, py, z) = pt.project(
                    cos_lat * clon * rr,
                    sin_lat * rr,
                    cos_lat * slon * rr,
                );
                let depth = f32::mul_add(z, inv_r, 1.0) / 2.0;
                dots.push(Dot::new(
                    px,
                    py,
                    z,
                    r_depth.mul_add(depth, r_base) * crest_r * rs,
                    0.56f32.mul_add(-depth, 0.66) - crest_ink,
                ));
            }
        });
    }
}
