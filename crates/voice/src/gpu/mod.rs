//! Optional GPU backend for the nemotron encoder (feature `voice/gpu`).
//!
//! Everything in this module is compiled only when the `gpu` feature is
//! enabled and is never required at runtime: if `GpuContext::init()`
//! returns `None`, the voice crate keeps using the CPU path untouched.

pub mod context;
pub mod encoder;
pub mod kernels;
pub mod model;
pub mod streaming;

pub use context::GpuContext;
