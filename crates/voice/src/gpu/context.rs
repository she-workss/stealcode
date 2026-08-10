//! Cross-platform GPU compute context (wgpu: Vulkan/DX12/Metal).
//!
//! This is the shared entry point for the voice GPU backend. It owns the
//! device/queue and a few helpers for building buffers and pipelines and
//! for downloading results back to the CPU. Everything here is strictly
//! optional — the CPU path never touches this module.

use std::sync::Arc;

use anyhow::Result;
use tracing::info;

/// A thin wrapper over a wgpu device/queue plus the small helpers the
/// kernels below need (buffers, pipelines, staged download).
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    /// True when `enable f16;` is usable in shaders.
    pub f16: bool,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .field("f16", &self.f16)
            .finish()
    }
}

impl GpuContext {
    /// Try to initialize a GPU context on any available backend
    /// (Vulkan, DX12 or Metal depending on the OS). Returns `None` when
    /// no usable adapter/device exists so callers can fall back to CPU.
    ///
    /// Uses a high-performance (discrete) adapter when present. f16
    /// (`Features::SHADER_F16`) is requested opportunistically — if the
    /// adapter lacks it the context is still created and `f16` is false.
    pub fn init() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: true,
            },
        ))
        .ok()?;
        let adapter_info = adapter.get_info();
        let mut features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::SHADER_F16)
        {
            features |= wgpu::Features::SHADER_F16;
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("stealcode-voice-gpu"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            },
        ))
        .ok()?;
        let f16 = features.contains(wgpu::Features::SHADER_F16);
        info!(
            "voice/gpu: adapter={} backend={:?} f16={f16}",
            adapter_info.name, adapter_info.backend
        );
        Some(GpuContext {
            device: Arc::new(device),
            queue,
            adapter_info,
            features,
            limits,
            f16,
        })
    }

    pub fn supports_f16(&self) -> bool {
        self.f16
    }

    /// Compile a WGSL module (fails loudly on shader syntax errors so
    /// they surface during development, not silently).
    pub fn shader(&self, label: &str, source: &str) -> Result<wgpu::ShaderModule> {
        Ok(self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            }))
    }

    /// Build a compute pipeline from a module and bind-group layout.
    pub fn pipeline(
        &self,
        label: &str,
        module: &wgpu::ShaderModule,
        layout: &wgpu::BindGroupLayout,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Create a GPU buffer for compute usage (bytes -> storage).
    pub fn storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(label, size, wgpu::BufferUsages::STORAGE)
    }

    /// Create a GPU buffer usable as a compute input (readable by
    /// shaders) and as a copy destination.
    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usages: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usages,
            mapped_at_creation: false,
        })
    }

    /// Upload `contents` into a new buffer with `usage`.
    ///
    /// Uses `queue.write_buffer` (backed by a per-call staging buffer that is
    /// freed on the next submission). This keeps a CPU copy of `contents`
    /// transiently, which is fine for small buffers. For bulk weight uploads
    /// use a `StagingBelt` (see `GpuModel::from_encoder`) so staging memory
    /// is reused instead of doubling the working set.
    pub fn upload(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, contents);
        buf
    }

    /// Copy `size` bytes from `src` into a CPU-visible staging buffer
    /// and map them back, blocking until the copy is done. Used for
    /// parity checks and for the final encoder/decoder readback.
    pub fn download(&self, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voice/gpu download staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu download"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("voice/gpu download poll");
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("voice/gpu map poll");
        let _ = rx.recv();
        let data = slice
            .get_mapped_range()
            .expect("voice/gpu map range")
            .to_vec();
        staging.unmap();
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_init_and_download() {
        let ctx = GpuContext::init().expect("no GPU adapter available");
        // Round-trip a small buffer through the GPU to exercise the
        // upload/download paths.
        let data: [u8; 64] = (0..64u8).collect::<Vec<_>>().try_into().unwrap();
        let buf = ctx.upload(
            "test",
            &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let back = ctx.download(&buf, data.len() as u64);
        assert_eq!(back, data);
        info!("gpu init ok: {:?}", ctx.adapter_info);
    }
}
