//! # gpui-thinking-orbs
//!
//! Dotted thought-orb loading indicators for AI & agent UIs - a native
//! [GPUI](https://www.gpui.rs/) port of
//! [thinking-orbs](https://orbs.jakubantalik.com/) by Jakub Antalik.
//!
//! Twelve hand-tuned animated states, four size presets (inline 20 / avatar 64
//! / large 96 / hero 128), monochrome light/dark ink on a transparent canvas.
//! The animation math is backend-agnostic; only the paint path talks to GPUI.
//!
//! ## Quick start
//!
//! ```ignore
//! use gpui_thinking_orbs::{OrbState, ThinkingOrb};
//!
//! // Inside a GPUI app:
//! cx.open_window(options, |_, cx| {
//!     cx.new(|_| ThinkingOrb::new().state(OrbState::Searching))
//! });
//! ```
//!
//! ## API stability
//!
//! **Supported app API:** [`ThinkingOrb`], [`OrbState`], [`OrbSize`],
//! [`OrbTheme`], [`resolve_preset`]. Prefer these in application code.
//!
//! **Advanced:** [`draw_mode`], [`ModeOpts`], [`Frame`], etc. - useful for
//! custom paint loops; may evolve more freely while the crate is `0.x`.
//! See the crate README ("API surface") and `CHANGELOG.md`.
//!
//! ## States
//!
//! | State | Look |
//! |-------|------|
//! | `Working` | particles on tilted orbits |
//! | `Searching` | scan meridian on a dotted globe |
//! | `Solving` | Rubik-style band scramble → solve |
//! | `Listening` | waveform through latitude rings |
//! | `Connecting` | constellation + packet edges |
//! | `Weaving` | three strands plaiting |
//! | `Composing` | undulating multi-band sash |
//! | `Breathing` | face-on ring morph |
//! | `Shaping` | outline circle → triangle → square |
//! | `Focusing` | particle iris converging on a focal core |
//! | `Reasoning` | counter-rotating gyroscope loops |
//! | `Recalling` | memory echoes expanding from a core |
//!
//! Animation geometry is pure Rust (no DOM, no `WebGL`). Dots are painted as
//! rounded GPUI quads; constellation edges use stroked paths.

mod engine;
mod paint;
mod presets;
mod types;
mod widget;

pub use engine::{
    Dot, Frame, Line, MAX_ICON_D, MAX_MORPH_DOTS, MAX_NODE_N, MAX_SIZE,
    MIN_SIZE, ModeOpts, Proj, base_profile, draw_mode, draw_mode_into,
    make_proj, sanitize_mode_opts, sanitize_size, scale_counts, scale_radii,
    sort_dots,
};
pub use presets::{Resolved, resolve_preset};
pub use types::{ModeKey, OrbSize, OrbState, OrbTheme};
pub use widget::{DEFAULT_TARGET_FPS, ThinkingOrb, thinking_orb};
