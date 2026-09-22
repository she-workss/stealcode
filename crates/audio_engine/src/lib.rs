//! Pure-Rust streaming speech-to-text engine (Nemotron RNNT), with an
//! optional wgpu GPU backend for the encoder. The app-facing `voice`
//! service builds on the [`model`] traits.

#![feature(portable_simd)]
#![feature(core_intrinsics)]
// `core_intrinsics::prefetch_read_data` is used deliberately by the
// software-prefetching SIMD kernels (`simd_kernel::prefetch_next_row`); the
// nightly feature is required for it and the lint is not actionable here.
#![allow(internal_features)]

pub mod dsp;
pub mod gguf;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod math;
pub mod model;
pub mod nemotron;
mod pool;
pub mod sgemm_kernel;
pub mod simd_kernel;
pub mod streaming;
pub mod tokenizer;

pub use nemotron::Nemotron;
