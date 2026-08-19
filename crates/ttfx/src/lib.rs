#![feature(random)]

//! ttfx: terminal text effects engine.
//!
//! Build an effect by name and drive it through an [`EngineCtx`], or step it
//! frame by frame and read cells via [`Terminal::frame_cells`] to render with
//! any backend (see `examples/all.rs`).
//!
//! [`EngineCtx`]: engine::ctx::EngineCtx
//! [`Terminal::frame_cells`]: engine::terminal::Terminal::frame_cells

pub mod effects;
pub mod engine;
pub mod utils;
