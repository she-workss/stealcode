//! Easing functions, ported from utils/easing.py.
//!
//! Transcription rule: Python `x ** n` routes through C `pow()` even for int
//! exponents, so every `**` here is `powf`, never `powi` - they can differ by
//! ULPs and coordinate quantization sits downstream (plan.md §5.20).

use std::f64::consts::PI;

/// A named easing or a custom cubic bezier (`make_easing`). Copyable so Paths
/// and Scenes can carry it by value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Easing {
    Linear,
    InSine,
    OutSine,
    InOutSine,
    InQuad,
    OutQuad,
    InOutQuad,
    InCubic,
    OutCubic,
    InOutCubic,
    InQuart,
    OutQuart,
    InOutQuart,
    InQuint,
    OutQuint,
    InOutQuint,
    InExpo,
    OutExpo,
    InOutExpo,
    InCirc,
    OutCirc,
    InOutCirc,
    InBack,
    OutBack,
    InOutBack,
    InElastic,
    OutElastic,
    InOutElastic,
    InBounce,
    OutBounce,
    InOutBounce,
    /// `easing.make_easing(x1`, y1, x2, y2): CSS-style cubic bezier.
    CubicBezier(f64, f64, f64, f64),
}

impl Easing {
    /// CLI parser for the 31 named functions (`argutils.Ease.type_parser`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_lowercase().as_str() {
            "linear" => Self::Linear,
            "in_sine" => Self::InSine,
            "out_sine" => Self::OutSine,
            "in_out_sine" => Self::InOutSine,
            "in_quad" => Self::InQuad,
            "out_quad" => Self::OutQuad,
            "in_out_quad" => Self::InOutQuad,
            "in_cubic" => Self::InCubic,
            "out_cubic" => Self::OutCubic,
            "in_out_cubic" => Self::InOutCubic,
            "in_quart" => Self::InQuart,
            "out_quart" => Self::OutQuart,
            "in_out_quart" => Self::InOutQuart,
            "in_quint" => Self::InQuint,
            "out_quint" => Self::OutQuint,
            "in_out_quint" => Self::InOutQuint,
            "in_expo" => Self::InExpo,
            "out_expo" => Self::OutExpo,
            "in_out_expo" => Self::InOutExpo,
            "in_circ" => Self::InCirc,
            "out_circ" => Self::OutCirc,
            "in_out_circ" => Self::InOutCirc,
            "in_back" => Self::InBack,
            "out_back" => Self::OutBack,
            "in_out_back" => Self::InOutBack,
            "in_elastic" => Self::InElastic,
            "out_elastic" => Self::OutElastic,
            "in_out_elastic" => Self::InOutElastic,
            "in_bounce" => Self::InBounce,
            "out_bounce" => Self::OutBounce,
            "in_out_bounce" => Self::InOutBounce,
            _ => return None,
        })
    }

    #[must_use]
    #[allow(clippy::float_cmp)] // endpoint checks: easing is exact at p == 0.0 / 1.0
    pub fn ease(&self, p: f64) -> f64 {
        match *self {
            Self::Linear => p,
            Self::InSine => 1.0 - ((p * PI) / 2.0).cos(),
            Self::OutSine => ((p * PI) / 2.0).sin(),
            Self::InOutSine => -((PI * p).cos() - 1.0) / 2.0,
            Self::InQuad => p.powi(2),
            Self::OutQuad => (1.0 - p).mul_add(-(1.0 - p), 1.0),
            Self::InOutQuad => {
                if p < 0.5 {
                    2.0 * p.powi(2)
                } else {
                    1.0 - (-2.0f64).mul_add(p, 2.0).powi(2) / 2.0
                }
            }
            Self::InCubic => p.powi(3),
            Self::OutCubic => 1.0 - (1.0 - p).powi(3),
            Self::InOutCubic => {
                if p < 0.5 {
                    4.0 * p.powi(3)
                } else {
                    1.0 - (-2.0f64).mul_add(p, 2.0).powi(3) / 2.0
                }
            }
            Self::InQuart => p.powi(4),
            Self::OutQuart => 1.0 - (1.0 - p).powi(4),
            Self::InOutQuart => {
                if p < 0.5 {
                    8.0 * p.powi(4)
                } else {
                    1.0 - (-2.0f64).mul_add(p, 2.0).powi(4) / 2.0
                }
            }
            Self::InQuint => p.powi(5),
            Self::OutQuint => 1.0 - (1.0 - p).powi(5),
            Self::InOutQuint => {
                if p < 0.5 {
                    16.0 * p.powi(5)
                } else {
                    1.0 - (-2.0f64).mul_add(p, 2.0).powi(5) / 2.0
                }
            }
            Self::InExpo => {
                if p == 0.0 {
                    0.0
                } else {
                    10.0f64.mul_add(p, -10.0).exp2()
                }
            }
            Self::OutExpo => {
                if p == 1.0 {
                    1.0
                } else {
                    1.0 - (-10.0 * p).exp2()
                }
            }
            Self::InOutExpo => {
                if p == 0.0 {
                    0.0
                } else if p == 1.0 {
                    1.0
                } else if p < 0.5 {
                    20.0f64.mul_add(p, -10.0).exp2() / 2.0
                } else {
                    (2.0 - (-20.0f64).mul_add(p, 10.0).exp2()) / 2.0
                }
            }
            Self::InCirc => 1.0 - p.mul_add(-p, 1.0).sqrt(),
            Self::OutCirc => (p - 1.0).mul_add(-(p - 1.0), 1.0).sqrt(),
            Self::InOutCirc => {
                if p < 0.5 {
                    (1.0 - (2.0 * p).mul_add(-(2.0 * p), 1.0).sqrt()) / 2.0
                } else {
                    f64::midpoint(
                        (-2.0f64)
                            .mul_add(p, 2.0)
                            .mul_add(-(-2.0f64).mul_add(p, 2.0), 1.0)
                            .sqrt(),
                        1.0,
                    )
                }
            }
            Self::InBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                f64::mul_add(c1, -p.powi(2), c3 * p.powi(3))
            }
            Self::OutBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                f64::mul_add(
                    c1,
                    (p - 1.0).powi(2),
                    f64::mul_add(c3, (p - 1.0).powi(3), 1.0),
                )
            }
            Self::InOutBack => {
                let c1 = 1.70158;
                let c2 = c1 * 1.525;
                if p < 0.5 {
                    ((2.0 * p).powi(2) * f64::mul_add((c2 + 1.0) * 2.0, p, -c2))
                        / 2.0
                } else {
                    2.0f64.mul_add(p, -2.0).powi(2).mul_add(
                        f64::mul_add(c2 + 1.0, p.mul_add(2.0, -2.0), c2),
                        2.0,
                    ) / 2.0
                }
            }
            Self::InElastic => {
                let c4 = (2.0 * PI) / 3.0;
                if p == 0.0 {
                    0.0
                } else if p == 1.0 {
                    1.0
                } else {
                    -10.0f64.mul_add(p, -10.0).exp2()
                        * (p.mul_add(10.0, -10.75) * c4).sin()
                }
            }
            Self::OutElastic => {
                let c4 = (2.0 * PI) / 3.0;
                if p == 0.0 {
                    0.0
                } else if p == 1.0 {
                    1.0
                } else {
                    (-10.0 * p)
                        .exp2()
                        .mul_add((p.mul_add(10.0, -0.75) * c4).sin(), 1.0)
                }
            }
            Self::InOutElastic => {
                let c5 = (2.0 * PI) / 4.5;
                if p == 0.0 {
                    0.0
                } else if p == 1.0 {
                    1.0
                } else if p < 0.5 {
                    -(20.0f64.mul_add(p, -10.0).exp2()
                        * (20.0f64.mul_add(p, -11.125) * c5).sin())
                        / 2.0
                } else {
                    ((-20.0f64).mul_add(p, 10.0).exp2()
                        * (20.0f64.mul_add(p, -11.125) * c5).sin())
                        / 2.0
                        + 1.0
                }
            }
            Self::InBounce => 1.0 - out_bounce(1.0 - p),
            Self::OutBounce => out_bounce(p),
            Self::InOutBounce => {
                if p < 0.5 {
                    (1.0 - out_bounce(2.0f64.mul_add(-p, 1.0))) / 2.0
                } else {
                    f64::midpoint(1.0, out_bounce(2.0f64.mul_add(p, -1.0)))
                }
            }
            Self::CubicBezier(x1, y1, x2, y2) => {
                bezier_easing(x1, y1, x2, y2, p)
            }
        }
    }
}

fn out_bounce(p: f64) -> f64 {
    let n1 = 7.5625;
    let d1 = 2.75;
    if p < 1.0 / d1 {
        n1 * p.powi(2)
    } else if p < 2.0 / d1 {
        f64::mul_add(n1, (p - 1.5 / d1).powi(2), 0.75)
    } else if p < 2.5 / d1 {
        f64::mul_add(n1, (p - 2.25 / d1).powi(2), 0.9375)
    } else {
        f64::mul_add(n1, (p - 2.625 / d1).powi(2), 0.984_375)
    }
}

/// `easing.make_easing`'s `bezier_easing`: Newton-Raphson on x with the exact
/// upstream constants (20 iterations, 1e-5 convergence, 1e-6 derivative bail).
/// Upstream `lru_caches` this; we just recompute (behavior-identical).
fn bezier_easing(x1: f64, y1: f64, x2: f64, y2: f64, progress: f64) -> f64 {
    let sample_curve_x = |t: f64| {
        (3.0 * x2 * (1.0 - t))
            .mul_add(t.powi(2), 3.0 * x1 * (1.0 - t).powi(2) * t)
            + t.powi(3)
    };
    let sample_curve_y = |t: f64| {
        (3.0 * y2 * (1.0 - t))
            .mul_add(t.powi(2), 3.0 * y1 * (1.0 - t).powi(2) * t)
            + t.powi(3)
    };
    let sample_curve_derivative_x = |t: f64| {
        (3.0 * t.powi(2)).mul_add(
            1.0 - x2,
            (6.0 * (1.0 - t) * t)
                .mul_add(x2 - x1, 3.0 * (1.0 - t).powi(2) * x1),
        )
    };

    if progress <= 0.0 {
        return 0.0;
    }
    if progress >= 1.0 {
        return 1.0;
    }
    let mut t = progress;
    for _ in 0..20 {
        let x_est = sample_curve_x(t);
        let dx = x_est - progress;
        if dx.abs() < 1e-5 {
            break;
        }
        let d = sample_curve_derivative_x(t);
        if d.abs() < 1e-6 {
            break;
        }
        t -= dx / d;
    }
    sample_curve_y(t)
}

/// easing.EasingTracker.
#[derive(Debug, Clone)]
pub struct EasingTracker {
    pub easing_function: Easing,
    pub total_steps: i64,
    pub current_step: i64,
    pub progress_ratio: f64,
    pub eased_value: f64,
}

impl EasingTracker {
    #[must_use]
    pub const fn new(easing_function: Easing, total_steps: i64) -> Self {
        Self {
            easing_function,
            total_steps,
            current_step: 0,
            progress_ratio: 0.0,
            eased_value: 0.0,
        }
    }

    pub fn step(&mut self) -> f64 {
        if self.current_step < self.total_steps {
            self.current_step += 1;
            self.progress_ratio =
                self.current_step as f64 / self.total_steps as f64;
            self.eased_value = self
                .easing_function
                .ease(self.progress_ratio)
                .clamp(0.0, 1.0);
        }
        self.eased_value
    }

    pub const fn reset(&mut self) {
        self.current_step = 0;
        self.progress_ratio = 0.0;
        self.eased_value = 0.0;
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

/// easing.SequenceEaser over owned elements (effects use it over id groups).
#[derive(Debug, Clone)]
pub struct SequenceEaser<T> {
    pub sequence: Vec<T>,
    pub easing_tracker: EasingTracker,
}

#[derive(Debug, Clone, Copy)]
pub struct SequenceStep<'a, T> {
    pub added: &'a [T],
    pub removed: &'a [T],
}

impl<T> SequenceEaser<T> {
    #[must_use]
    pub const fn new(
        sequence: Vec<T>,
        easing_function: Easing,
        total_steps: i64,
    ) -> Self {
        Self {
            sequence,
            easing_tracker: EasingTracker::new(easing_function, total_steps),
        }
    }

    pub fn step(&mut self) -> SequenceStep<'_, T> {
        let previous_eased = self.easing_tracker.eased_value;
        let eased_value = self.easing_tracker.step();
        let seq_len = self.sequence.len();
        if seq_len == 0 {
            return SequenceStep {
                added: &[],
                removed: &[],
            };
        }
        // int() truncation, faithfully
        let length = (eased_value * seq_len as f64) as i64 as usize;
        let previous_length = (previous_eased * seq_len as f64) as i64 as usize;

        match length.cmp(&previous_length) {
            std::cmp::Ordering::Greater => SequenceStep {
                added: &self.sequence[previous_length..length],
                removed: &[],
            },
            std::cmp::Ordering::Less => SequenceStep {
                added: &[],
                removed: &self.sequence[length..previous_length],
            },
            std::cmp::Ordering::Equal => SequenceStep {
                added: &[],
                removed: &[],
            },
        }
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.easing_tracker.is_complete()
    }

    pub const fn reset(&mut self) {
        self.easing_tracker.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::{Easing, SequenceEaser};

    #[derive(Debug, PartialEq)]
    struct NonClone(u8);

    #[test]
    fn sequence_easer_returns_borrowed_deltas_without_clone() {
        let mut easer = SequenceEaser::new(
            vec![NonClone(0), NonClone(1), NonClone(2), NonClone(3)],
            Easing::Linear,
            4,
        );

        assert_eq!(easer.step().added, &[NonClone(0)]);
        assert_eq!(easer.step().added, &[NonClone(1)]);
        assert_eq!(easer.step().added, &[NonClone(2)]);
        assert_eq!(easer.step().added, &[NonClone(3)]);
        assert_eq!(easer.step().added, []);

        easer.reset();
        assert_eq!(easer.step().added, &[NonClone(0)]);
    }

    #[test]
    fn sequence_easer_reports_reversed_easing_as_removed() {
        let sequence: Vec<usize> = (0..100).collect();
        let mut easer = SequenceEaser::new(sequence, Easing::OutBounce, 100);
        let mut saw_removed = false;

        for _ in 0..100 {
            let step = easer.step();
            assert!(step.added.is_empty() || step.removed.is_empty());
            saw_removed |= !step.removed.is_empty();
        }

        assert!(saw_removed);
    }
}
