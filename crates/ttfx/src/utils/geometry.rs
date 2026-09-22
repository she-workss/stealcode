//! Coord and geometry math, ported from utils/geometry.py.
//!
//! Upstream wraps every function in `lru_cache`; behavior is identical without
//! the caches, so they are omitted. All `round()` calls are banker's rounding
//! (pycompat), all `int()` casts truncate.

use rustc_hash::FxHashSet;

use crate::utils::pycompat::round_half_even;

/// 1-based canvas coordinate: column grows right, row grows UP (origin
/// bottom-left).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coord {
    pub column: i64,
    pub row: i64,
}

impl Coord {
    #[must_use]
    pub const fn new(column: i64, row: i64) -> Self {
        Self { column, row }
    }
}

/// Float-valued point for bezier intermediates: upstream builds Coord objects
/// with float fields inside `de_casteljau` (violating its own annotation) and
/// only rounds the final result.
#[derive(Debug, Clone, Copy)]
struct FloatPoint {
    column: f64,
    row: f64,
}

impl FloatPoint {
    #[inline]
    fn interpolate(self, other: Self, t: f64) -> Self {
        Self {
            column: t.mul_add(other.column, (1.0 - t) * self.column),
            row: t.mul_add(other.row, (1.0 - t) * self.row),
        }
    }
}

/// `find_coords_on_circle`: `coords_limit` 0 -> round(2*pi*r); x offset from
/// the origin is doubled for cell aspect; every point rounded (banker's).
#[must_use]
pub fn find_coords_on_circle(
    origin: Coord,
    radius: i64,
    coords_limit: i64,
    unique: bool,
) -> Vec<Coord> {
    let mut points: Vec<Coord> = Vec::new();
    if radius == 0 {
        return points;
    }
    let mut seen: FxHashSet<Coord> = FxHashSet::default();
    let coords_limit = if coords_limit == 0 {
        round_half_even(2.0 * std::f64::consts::PI * radius as f64)
    } else {
        coords_limit
    };
    let angle_step = 2.0 * std::f64::consts::PI / coords_limit as f64;
    for i in 0..coords_limit {
        let angle = angle_step * i as f64;
        let mut x = (radius as f64).mul_add(angle.cos(), origin.column as f64);
        let x_diff = x - origin.column as f64;
        x += x_diff;
        let y = (radius as f64).mul_add(angle.sin(), origin.row as f64);
        let point = Coord::new(round_half_even(x), round_half_even(y));
        if unique {
            if !seen.contains(&point) {
                points.push(point);
            }
        } else {
            points.push(point);
        }
        seen.insert(point);
    }
    points
}

/// `find_coords_in_circle`: actually an ellipse (a = diameter, b = diameter/2);
/// `int()` truncation on the y offset, faithfully.
#[must_use]
pub fn find_coords_in_circle(center: Coord, diameter: i64) -> Vec<Coord> {
    let (h, k) = (center.column, center.row);
    let mut coords: Vec<Coord> = Vec::new();
    if diameter == 0 {
        return coords;
    }
    let a_squared = (diameter as f64).powi(2);
    let b_squared = (diameter as f64 / 2.0).powi(2);
    for x in (h - diameter)..=(h + diameter) {
        let x_component = ((x - h) as f64).powi(2) / a_squared;
        let max_y_offset = (b_squared * (1.0 - x_component)).sqrt() as i64;
        for y in (k - max_y_offset)..=(k + max_y_offset) {
            coords.push(Coord::new(x, y));
        }
    }
    coords
}

/// `find_coords_in_rect`: full (2d+1)^2 block, empty for distance 0.
/// Iteration order is column-major like upstream.
#[must_use]
pub fn find_coords_in_rect(origin: Coord, distance: i64) -> Vec<Coord> {
    let mut coords: Vec<Coord> = Vec::new();
    if distance == 0 {
        return coords;
    }
    for column in (origin.column - distance)..=(origin.column + distance) {
        for row in (origin.row - distance)..=(origin.row + distance) {
            coords.push(Coord::new(column, row));
        }
    }
    coords
}

/// `find_coords_on_rect`: perimeter only; empty if either half-dimension is 0.
#[must_use]
pub fn find_coords_on_rect(
    origin: Coord,
    half_width: i64,
    half_height: i64,
) -> Vec<Coord> {
    let mut coords: Vec<Coord> = Vec::new();
    if half_width == 0 || half_height == 0 {
        return coords;
    }
    for column in (origin.column - half_width)..=(origin.column + half_width) {
        if column == origin.column - half_width
            || column == origin.column + half_width
        {
            for row in (origin.row - half_height)..=(origin.row + half_height) {
                coords.push(Coord::new(column, row));
            }
        } else {
            coords.push(Coord::new(column, origin.row - half_height));
            coords.push(Coord::new(column, origin.row + half_height));
        }
    }
    coords
}

/// `extrapolate_along_ray`: NON-doubled line length, lerp past the target,
/// round.
#[must_use]
pub fn extrapolate_along_ray(
    origin: Coord,
    target: Coord,
    offset_from_target: f64,
) -> Coord {
    let base = find_length_of_line(origin, target, false);
    let total_distance = base + offset_from_target;
    if total_distance == 0.0 || origin == target {
        return target;
    }
    let t = total_distance / base;
    let next_column =
        t.mul_add(target.column as f64, (1.0 - t) * origin.column as f64);
    let next_row = t.mul_add(target.row as f64, (1.0 - t) * origin.row as f64);
    Coord::new(round_half_even(next_column), round_half_even(next_row))
}

/// `find_coord_on_bezier_curve`: recursive De Casteljau of arbitrary degree
/// with float intermediates, rounded only at the end.
#[must_use]
pub fn find_coord_on_bezier_curve(
    start: Coord,
    control: &[Coord],
    end: Coord,
    t: f64,
) -> Coord {
    if control.is_empty() {
        return find_coord_on_line(start, end, t);
    }

    let start = FloatPoint {
        column: start.column as f64,
        row: start.row as f64,
    };
    let end = FloatPoint {
        column: end.column as f64,
        row: end.row as f64,
    };

    // Every production path is quadratic. Keep that per-frame hot path on the
    // stack instead of allocating a Vec at each De Casteljau level.
    if let [control] = control {
        let control = FloatPoint {
            column: control.column as f64,
            row: control.row as f64,
        };
        let point = start
            .interpolate(control, t)
            .interpolate(control.interpolate(end, t), t);
        return Coord::new(
            round_half_even(point.column),
            round_half_even(point.row),
        );
    }

    let mut points: Vec<FloatPoint> = Vec::with_capacity(control.len() + 2);
    points.push(start);
    for c in control {
        points.push(FloatPoint {
            column: c.column as f64,
            row: c.row as f64,
        });
    }
    points.push(end);
    let mut remaining = points.len();
    while remaining > 1 {
        for i in 0..remaining - 1 {
            points[i] = points[i].interpolate(points[i + 1], t);
        }
        remaining -= 1;
    }
    Coord::new(
        round_half_even(points[0].column),
        round_half_even(points[0].row),
    )
}

/// `find_coord_on_line`: lerp + round.
#[must_use]
pub fn find_coord_on_line(start: Coord, end: Coord, t: f64) -> Coord {
    let x = t.mul_add(end.column as f64, (1.0 - t) * start.column as f64);
    let y = t.mul_add(end.row as f64, (1.0 - t) * start.row as f64);
    Coord::new(round_half_even(x), round_half_even(y))
}

/// `find_length_of_bezier_curve`: 10-sample polyline that stops at t=0.9 - the
/// final t=0.9..1.0 span is deliberately (faithfully) omitted, systematically
/// underestimating lengths. Do not fix (plan.md §5.4).
#[must_use]
pub fn find_length_of_bezier_curve(
    start: Coord,
    control: &[Coord],
    end: Coord,
) -> f64 {
    let mut length = 0.0;
    let mut prev_coord = start;
    for t in 1..10 {
        let coord =
            find_coord_on_bezier_curve(start, control, end, t as f64 / 10.0);
        length += find_length_of_line(prev_coord, coord, true);
        prev_coord = coord;
    }
    length
}

/// `find_length_of_line`: hypot, with the row delta doubled when requested
/// (terminal cell aspect convention).
#[must_use]
pub fn find_length_of_line(
    coord1: Coord,
    coord2: Coord,
    double_row_diff: bool,
) -> f64 {
    let column_diff = (coord2.column - coord1.column) as f64;
    let row_diff = (coord2.row - coord1.row) as f64;
    if double_row_diff {
        f64::hypot(column_diff, 2.0 * row_diff)
    } else {
        f64::hypot(column_diff, row_diff)
    }
}

/// `find_normalized_distance_from_center`: rejects out-of-rectangle coords
/// (upstream `ValueError`); stays within [0, 1] for accepted ones.
pub fn find_normalized_distance_from_center(
    bottom: i64,
    top: i64,
    left: i64,
    right: i64,
    other_coord: Coord,
) -> Result<f64, String> {
    let y_offset = bottom - 1;
    let x_offset = left - 1;
    let right = right - x_offset;
    let top = top - y_offset;
    let center_x = right as f64 / 2.0;
    let center_y = top as f64 / 2.0;

    // Python: `n not in range(a, b+1)` - integer membership
    let col = other_coord.column - x_offset;
    let row = other_coord.row - y_offset;
    if !(left - x_offset..=right).contains(&col)
        || !(bottom - y_offset..=top).contains(&row)
    {
        return Err("Coordinate is not within the rectangle.".to_string());
    }

    let max_distance = ((top * 2) as f64)
        .mul_add((top * 2) as f64, (right as f64).powi(2))
        .sqrt();
    let distance = ((row as f64 - center_y) * 2.0)
        .mul_add(
            (row as f64 - center_y) * 2.0,
            (col as f64 - center_x).powi(2),
        )
        .sqrt();
    Ok(distance / (max_distance / 2.0))
}
