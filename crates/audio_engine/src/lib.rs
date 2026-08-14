//! Pure-Rust streaming speech-to-text engine (Nemotron RNNT), with an
//! optional wgpu GPU backend for the encoder. The app-facing `voice`
//! service builds on the [`model`] traits.

pub mod dsp;
pub mod gguf;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod math;
pub mod model;
pub mod nemotron;
pub mod sgemm_kernel;
pub mod streaming;
pub mod tokenizer;

pub use nemotron::Nemotron;
