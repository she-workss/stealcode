//! Host-side wrappers for the WGSL compute kernels.
//!
//! Kernel shaders live in `shaders/*.wgsl`. Each wrapper owns the
//! pipeline/bind-group-layout for its kernel; buffers that persist
//! across calls (model weights) are owned by the caller.

use std::sync::Arc;

use anyhow::Result;

use super::context::GpuContext;
use crate::nemotron::gguf::f16_to_f32;

const Q8_GEMM_WGSL: &str = include_str!("shaders/q8_gemm.wgsl");

/// A matrix packed for the GPU Q8 GEMM: per output row `kb` blocks of
/// 32 int8 weights, 4 per `u32` in `q`, one f32 scale per block in `s`.
pub struct PackedQ8 {
    /// `rows * kb * 8` u32, little-endian.
    pub q: Vec<u8>,
    /// `rows * kb` f32, little-endian.
    pub s: Vec<u8>,
    /// Output rows of the weight matrix.
    pub rows: usize,
    /// Blocks per row (`k.div_ceil(32)`).
    pub kb: usize,
}

/// Pack raw Q8 bytes (one row per `padded_row` bytes, `block_bytes`
/// 34 for Q8F16 or 36 for Q8_0, scale first) into the GPU layout.
/// `rows` = weight output rows, `k` = weight input cols.
pub fn pack_q8(
    raw: &[u8],
    rows: usize,
    k: usize,
    padded_row: usize,
    block_bytes: usize,
) -> PackedQ8 {
    let kb = k.div_ceil(32);
    let mut q = Vec::with_capacity(rows * kb * 8 * 4);
    let mut s = Vec::with_capacity(rows * kb * 4);
    for r in 0..rows {
        let row = &raw[r * padded_row..(r + 1) * padded_row];
        for b in 0..kb {
            let block = &row[b * block_bytes..(b + 1) * block_bytes];
            let scale = if block_bytes == 34 {
                f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
            } else {
                f32::from_le_bytes([block[0], block[1], block[2], block[3]])
            };
            s.extend_from_slice(&scale.to_le_bytes());
            // 32 int8 values -> 8 u32.
            for i in 0..8 {
                let off = 2 + i * 4;
                let u = u32::from_le_bytes([
                    block[off] as i8 as u8,
                    block[off + 1] as i8 as u8,
                    block[off + 2] as i8 as u8,
                    block[off + 3] as i8 as u8,
                ]);
                q.extend_from_slice(&u.to_le_bytes());
            }
        }
    }
    PackedQ8 { q, s, rows, kb }
}

/// CPU reference for the Q8 GEMM: `y[t, n] = x[t, k] @ W_q8[n, k]^T`
/// with per-32-block f32 scaling (no activation quantization). Used for
/// parity tests against the GPU kernel.
pub fn q8_gemm_ref(
    packed: &PackedQ8,
    x: &[f32],
    t: usize,
    k: usize,
    y: &mut [f32],
) {
    let rows = packed.rows;
    let kb = packed.kb;
    // Weight value at output row `n`, k-position `j` (0..k).
    let wval = |n: usize, j: usize| -> i32 {
        let b = j / 32;
        let i = j % 32;
        let word = i / 4;
        let lane = i % 4;
        let u = u32::from_le_bytes([
            packed.q[(n * kb + b) * 32 + word * 4],
            packed.q[(n * kb + b) * 32 + word * 4 + 1],
            packed.q[(n * kb + b) * 32 + word * 4 + 2],
            packed.q[(n * kb + b) * 32 + word * 4 + 3],
        ]);
        let byte = (u >> (lane * 8)) & 0xff;
        byte as u8 as i8 as i32
    };
    for ti in 0..t {
        for n in 0..rows {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let scale = f32::from_le_bytes([
                    packed.s[(n * kb + b) * 4],
                    packed.s[(n * kb + b) * 4 + 1],
                    packed.s[(n * kb + b) * 4 + 2],
                    packed.s[(n * kb + b) * 4 + 3],
                ]);
                let mut bdot = 0.0f32;
                for i in 0..32 {
                    bdot += wval(n, b * 32 + i) as f32
                        * x[ti * k + b * 32 + i];
                }
                acc += scale * bdot;
            }
            y[ti * rows + n] = acc;
        }
    }
}

/// Compute pipeline for the packed-Q8 GEMM.
pub struct Q8Gemm {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    bias_dummy: wgpu::Buffer,
}

impl Q8Gemm {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu q8_gemm bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, true),
                    bind_buffer(4, true),
                    bind_buffer(5, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu q8_gemm shader", Q8_GEMM_WGSL)?;
        let pipeline = ctx.pipeline("voice/gpu q8_gemm", &module, &layout, "main");
        let params = ctx.create_buffer(
            "voice/gpu q8_gemm params",
            32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let x_buf = ctx.create_buffer(
            "voice/gpu q8_gemm x",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out_buf = ctx.create_buffer(
            "voice/gpu q8_gemm out",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let bias_dummy = ctx.storage_buffer("voice/gpu q8_gemm bias_dummy", 4);
        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            layout,
            params,
            x_buf,
            out_buf,
            bias_dummy,
        })
    }

    /// `y[t, n] = x[t, k] @ W^T + bias` where `W` is the packed Q8 weight
    /// matrix (n = packed.rows). Blocks until done and returns `y`.
    pub fn gemm(
        &mut self,
        packed: &PackedQ8,
        w_q: &wgpu::Buffer,
        w_s: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        x: &[f32],
        t: usize,
        k: usize,
    ) -> Vec<f32> {
        let n = packed.rows;
        let kb = packed.kb;
        let x_size = (t * k * 4).max(4) as u64;
        let out_size = (t * n * 4).max(4) as u64;
        if self.x_buf.size() < x_size {
            self.x_buf = self.ctx.create_buffer(
                "voice/gpu q8_gemm x",
                x_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        if self.out_buf.size() < out_size {
            self.out_buf = self.ctx.create_buffer(
                "voice/gpu q8_gemm out",
                out_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
        }
        let params: [u32; 8] = [
            t as u32,
            k as u32,
            n as u32,
            kb as u32,
            u32::from(bias.is_some()),
            0,
            0,
            0,
        ];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu q8_gemm"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.x_buf, 0, bytemuck_safe(x));
        let bias_buf = bias.unwrap_or(&self.bias_dummy);
        let bind = self.ctx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("voice/gpu q8_gemm bind"),
                layout: &self.layout,
                entries: &[
                    binding(0, &self.params),
                    binding(1, w_q),
                    binding(2, w_s),
                    binding(3, &self.x_buf),
                    binding(4, bias_buf),
                    binding(5, &self.out_buf),
                ],
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu q8_gemm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(t.div_ceil(8) as u32, n.div_ceil(8) as u32, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, out_size);
        bytes_to_f32(&bytes, t * n)
    }
}

const LAYERNORM_WGSL: &str = include_str!("shaders/layernorm.wgsl");
const ELEMENTWISE_WGSL: &str = include_str!("shaders/elementwise.wgsl");
const ATTENTION_WGSL: &str = include_str!("shaders/attention.wgsl");
const DWCONV_WGSL: &str = include_str!("shaders/dwconv.wgsl");

/// Ensure `buf` is at least `size` bytes, recreating it when it is not.
fn grow_buffer(
    ctx: &Arc<GpuContext>,
    buf: &mut wgpu::Buffer,
    label: &str,
    size: u64,
    usages: wgpu::BufferUsages,
) {
    if buf.size() < size {
        *buf = ctx.create_buffer(label, size, usages);
    }
}

/// Compute LayerNorm over rows: `out[t, d] = (x - mean) * rstd * gamma + beta`.
pub struct LayerNormKernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    gamma_buf: wgpu::Buffer,
    beta_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
}

impl LayerNormKernel {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu layernorm bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, true),
                    bind_buffer(4, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu layernorm shader", LAYERNORM_WGSL)?;
        let pipeline = ctx.pipeline("voice/gpu layernorm", &module, &layout, "main");
        let mk = |label: &str, usages| ctx.create_buffer(label, 4, usages);
        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            layout,
            params: ctx.create_buffer(
                "voice/gpu layernorm params",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            x_buf: mk(
                "voice/gpu layernorm x",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            gamma_buf: mk(
                "voice/gpu layernorm gamma",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            beta_buf: mk(
                "voice/gpu layernorm beta",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            out_buf: mk(
                "voice/gpu layernorm out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        })
    }

    /// `out[t, d] = (x[t, d] - mean) / sqrt(var + eps) * gamma[d] + beta[d]`.
    pub fn forward(
        &mut self,
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        t: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32> {
        let count = t * d;
        let size = (count * 4).max(4) as u64;
        let dsize = (d * 4).max(4) as u64;
        grow_buffer(
            &self.ctx,
            &mut self.x_buf,
            "voice/gpu layernorm x",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.gamma_buf,
            "voice/gpu layernorm gamma",
            dsize,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.beta_buf,
            "voice/gpu layernorm beta",
            dsize,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.out_buf,
            "voice/gpu layernorm out",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params: [u32; 4] = [t as u32, d as u32, eps.to_bits(), 0];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu layernorm"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.x_buf, 0, bytemuck_safe(x));
        self.ctx.queue.write_buffer(&self.gamma_buf, 0, bytemuck_safe(gamma));
        self.ctx.queue.write_buffer(&self.beta_buf, 0, bytemuck_safe(beta));
        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voice/gpu layernorm bind"),
            layout: &self.layout,
            entries: &[
                binding(0, &self.params),
                binding(1, &self.x_buf),
                binding(2, &self.gamma_buf),
                binding(3, &self.beta_buf),
                binding(4, &self.out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu layernorm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(t.div_ceil(64) as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, size);
        bytes_to_f32(&bytes, count)
    }
}

/// Elementwise activation / bias / residual kernels.
pub struct ElementwiseKernel {
    ctx: Arc<GpuContext>,
    silu: wgpu::ComputePipeline,
    relu: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    add_mul: wgpu::ComputePipeline,
    bias_add: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    dummy: wgpu::Buffer,
}

impl ElementwiseKernel {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu elementwise bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu elementwise shader", ELEMENTWISE_WGSL)?;
        let mk = |op: &str| ctx.pipeline(&format!("voice/gpu elementwise {op}"), &module, &layout, op);
        Ok(Self {
            ctx: ctx.clone(),
            silu: mk("silu"),
            relu: mk("relu"),
            glu: mk("glu"),
            add_mul: mk("add_mul"),
            bias_add: mk("bias_add"),
            layout,
            params: ctx.create_buffer(
                "voice/gpu elementwise params",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            x_buf: ctx.create_buffer(
                "voice/gpu elementwise x",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            y_buf: ctx.create_buffer(
                "voice/gpu elementwise y",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            out_buf: ctx.create_buffer(
                "voice/gpu elementwise out",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            dummy: ctx.storage_buffer("voice/gpu elementwise dummy", 4),
        })
    }

    fn run(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        x: &[f32],
        y: Option<&[f32]>,
        count: usize,
        dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let size = (count * 4).max(4) as u64;
        let x_size = (x.len() * 4).max(4) as u64;
        let y_size = (y.map_or(0, |v| v.len()) * 4).max(4) as u64;
        grow_buffer(
            &self.ctx,
            &mut self.x_buf,
            "voice/gpu elementwise x",
            x_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.y_buf,
            "voice/gpu elementwise y",
            y_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.out_buf,
            "voice/gpu elementwise out",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params: [u32; 4] = [count as u32, dim as u32, scale.to_bits(), 0];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu elementwise"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.x_buf, 0, bytemuck_safe(x));
        if let Some(v) = y {
            self.ctx.queue.write_buffer(&self.y_buf, 0, bytemuck_safe(v));
        }
        let y_buf = if y.is_some() { &self.y_buf } else { &self.dummy };
        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voice/gpu elementwise bind"),
            layout: &self.layout,
            entries: &[
                binding(0, &self.params),
                binding(1, &self.x_buf),
                binding(2, y_buf),
                binding(3, &self.out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu elementwise pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(256) as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, size);
        bytes_to_f32(&bytes, count)
    }

    /// `out = x * sigmoid(x)`, elementwise.
    pub fn silu(&mut self, x: &[f32]) -> Vec<f32> {
        let p = self.silu.clone();
        self.run(&p, x, None, x.len(), 0, 0.0)
    }

    /// `out = max(0, x)`, elementwise.
    pub fn relu(&mut self, x: &[f32]) -> Vec<f32> {
        let p = self.relu.clone();
        self.run(&p, x, None, x.len(), 0, 0.0)
    }

    /// `out[i] = x[i] * sigmoid(x[dim + i])`: GLU over a `[t, 2*dim]`
    /// input laid out `[gate block | value block]`. Returns `t*dim` values.
    pub fn glu(&mut self, x: &[f32], dim: usize) -> Vec<f32> {
        let count = x.len() / 2;
        let p = self.glu.clone();
        self.run(&p, x, None, count, dim, 0.0)
    }

    /// `out = a + scale * b`, elementwise.
    pub fn add_mul(&mut self, a: &[f32], b: &[f32], scale: f32) -> Vec<f32> {
        let p = self.add_mul.clone();
        self.run(&p, a, Some(b), a.len(), 0, scale)
    }

    /// `out[i] = x[i] + bias[i % bias.len()]`.
    pub fn bias_add(&mut self, x: &[f32], bias: &[f32]) -> Vec<f32> {
        let p = self.bias_add.clone();
        self.run(&p, x, Some(bias), x.len(), bias.len(), 0.0)
    }
}

/// Attention: relative-position biased scores, softmax over the band,
/// weighted V-sum. `head_dim` must equal the workgroup size (128).
pub struct AttentionKernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    qb_buf: wgpu::Buffer,
    vb_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
}

impl AttentionKernel {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu attention bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, true),
                    bind_buffer(4, true),
                    bind_buffer(5, true),
                    bind_buffer(6, true),
                    bind_buffer(7, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu attention shader", ATTENTION_WGSL)?;
        let pipeline = ctx.pipeline("voice/gpu attention", &module, &layout, "main");
        let mk = |label: &str, usages| ctx.create_buffer(label, 4, usages);
        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            layout,
            params: ctx.create_buffer(
                "voice/gpu attention params",
                32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            q_buf: mk("voice/gpu attention q", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            k_buf: mk("voice/gpu attention k", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            v_buf: mk("voice/gpu attention v", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            qb_buf: mk("voice/gpu attention q_bias", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            vb_buf: mk("voice/gpu attention v_bias", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            pos_buf: mk("voice/gpu attention pos", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            out_buf: mk(
                "voice/gpu attention out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        })
    }

    /// One workgroup per query frame; only supports `head_dim <= 128`.
    /// `left`/`right` define the attention band as in `EncoderConfig`
    /// (chunk_size = right + 1, left_chunks = left / chunk_size).
    pub fn forward(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_bias: &[f32],
        v_bias: &[f32],
        pos: &[f32],
        t: usize,
        d: usize,
        n_heads: usize,
        scale: f32,
        left: usize,
        right: usize,
    ) -> Vec<f32> {
        let count = t * d;
        let size = (count * 4).max(4) as u64;
        let dsize = (d * 4).max(4) as u64;
        let pos_size = (pos.len() * 4).max(4) as u64;
        for (buf, label, sz) in [
            (&mut self.q_buf, "voice/gpu attention q", size),
            (&mut self.k_buf, "voice/gpu attention k", size),
            (&mut self.v_buf, "voice/gpu attention v", size),
            (&mut self.qb_buf, "voice/gpu attention q_bias", dsize),
            (&mut self.vb_buf, "voice/gpu attention v_bias", dsize),
            (&mut self.pos_buf, "voice/gpu attention pos", pos_size),
        ] {
            grow_buffer(
                &self.ctx,
                buf,
                label,
                sz,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        grow_buffer(
            &self.ctx,
            &mut self.out_buf,
            "voice/gpu attention out",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let chunk_size = right + 1;
        let left_chunks = left / chunk_size;
        let params: [u32; 8] = [
            t as u32,
            d as u32,
            n_heads as u32,
            (d / n_heads) as u32,
            scale.to_bits(),
            0,
            chunk_size as u32,
            left_chunks as u32,
        ];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu attention"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.q_buf, 0, bytemuck_safe(q));
        self.ctx.queue.write_buffer(&self.k_buf, 0, bytemuck_safe(k));
        self.ctx.queue.write_buffer(&self.v_buf, 0, bytemuck_safe(v));
        self.ctx.queue.write_buffer(&self.qb_buf, 0, bytemuck_safe(q_bias));
        self.ctx.queue.write_buffer(&self.vb_buf, 0, bytemuck_safe(v_bias));
        self.ctx.queue.write_buffer(&self.pos_buf, 0, bytemuck_safe(pos));        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voice/gpu attention bind"),
            layout: &self.layout,
            entries: &[
                binding(0, &self.params),
                binding(1, &self.q_buf),
                binding(2, &self.k_buf),
                binding(3, &self.v_buf),
                binding(4, &self.qb_buf),
                binding(5, &self.vb_buf),
                binding(6, &self.pos_buf),
                binding(7, &self.out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu attention pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(t as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, size);
        bytes_to_f32(&bytes, count)
    }
}

const ATTN_STREAM_WGSL: &str = include_str!("shaders/attn_stream.wgsl");

/// Streaming attention: new query frames (offsets `s..s+c`) scored
/// against a combined band of frames `kv`/`vv` (absolute frames
/// `k_lo..k_hi`), with relative position rows `pos_p[qq - fr + pos_off]`.
/// One workgroup (128 lanes = head_dim) per new frame.
pub struct AttnStreamKernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    q_buf: wgpu::Buffer,
    kv_buf: wgpu::Buffer,
    vv_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    qb_buf: wgpu::Buffer,
    vb_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
}

impl AttnStreamKernel {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu attn_stream bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, true),
                    bind_buffer(4, true),
                    bind_buffer(5, true),
                    bind_buffer(6, true),
                    bind_buffer(7, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu attn_stream shader", ATTN_STREAM_WGSL)?;
        let pipeline = ctx.pipeline("voice/gpu attn_stream", &module, &layout, "main");
        let mk = |label: &str, usages| ctx.create_buffer(label, 4, usages);
        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            layout,
            params: ctx.create_buffer(
                "voice/gpu attn_stream params",
                48,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            q_buf: mk("voice/gpu attn_stream q", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            kv_buf: mk("voice/gpu attn_stream kv", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            vv_buf: mk("voice/gpu attn_stream vv", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            pos_buf: mk("voice/gpu attn_stream pos", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            qb_buf: mk("voice/gpu attn_stream q_bias", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            vb_buf: mk("voice/gpu attn_stream v_bias", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            out_buf: mk(
                "voice/gpu attn_stream out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        })
    }

    /// `out[qi, d]` for `qi in 0..c` new frames, absolute frame
    /// `qq = s + qi`. Band per query: chunk-aligned as in `EncoderConfig`,
    /// clamped to `[k_lo, k_hi)`. `pos_p` has `63` rows (rel -3..59).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        q: &[f32],
        kv: &[f32],
        vv: &[f32],
        pos_p: &[f32],
        q_bias: &[f32],
        v_bias: &[f32],
        c: usize,
        d: usize,
        n_heads: usize,
        scale: f32,
        s: usize,
        k_lo: usize,
        band: usize,
        chunk: usize,
        left_chunks: usize,
        k_hi: usize,
        pos_off: usize,
    ) -> Vec<f32> {
        let count = c * d;
        let size = (count * 4).max(4) as u64;
        let dsize = (d * 4).max(4) as u64;
        let band_size = (band * d * 4).max(4) as u64;
        let pos_size = (pos_p.len() * 4).max(4) as u64;
        for (buf, label, sz) in [
            (&mut self.q_buf, "voice/gpu attn_stream q", size),
            (&mut self.kv_buf, "voice/gpu attn_stream kv", band_size),
            (&mut self.vv_buf, "voice/gpu attn_stream vv", band_size),
            (&mut self.pos_buf, "voice/gpu attn_stream pos", pos_size),
            (&mut self.qb_buf, "voice/gpu attn_stream q_bias", dsize),
            (&mut self.vb_buf, "voice/gpu attn_stream v_bias", dsize),
        ] {
            grow_buffer(
                &self.ctx,
                buf,
                label,
                sz,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
        }
        grow_buffer(
            &self.ctx,
            &mut self.out_buf,
            "voice/gpu attn_stream out",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params: [u32; 12] = [
            c as u32,
            d as u32,
            n_heads as u32,
            (d / n_heads) as u32,
            scale.to_bits(),
            s as u32,
            k_lo as u32,
            band as u32,
            chunk as u32,
            left_chunks as u32,
            k_hi as u32,
            pos_off as u32,
        ];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu attn_stream"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.q_buf, 0, bytemuck_safe(q));
        self.ctx.queue.write_buffer(&self.kv_buf, 0, bytemuck_safe(kv));
        self.ctx.queue.write_buffer(&self.vv_buf, 0, bytemuck_safe(vv));
        self.ctx.queue.write_buffer(&self.pos_buf, 0, bytemuck_safe(pos_p));
        self.ctx.queue.write_buffer(&self.qb_buf, 0, bytemuck_safe(q_bias));
        self.ctx.queue.write_buffer(&self.vb_buf, 0, bytemuck_safe(v_bias));
        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voice/gpu attn_stream bind"),
            layout: &self.layout,
            entries: &[
                binding(0, &self.params),
                binding(1, &self.q_buf),
                binding(2, &self.kv_buf),
                binding(3, &self.vv_buf),
                binding(4, &self.pos_buf),
                binding(5, &self.qb_buf),
                binding(6, &self.vb_buf),
                binding(7, &self.out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu attn_stream pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(c as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, size);
        bytes_to_f32(&bytes, count)
    }
}

/// Causal depthwise 1-D convolution.
pub struct DwConvKernel {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    w_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
}

impl DwConvKernel {
    pub fn new(ctx: &Arc<GpuContext>) -> Result<Self> {
        let layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("voice/gpu dwconv bind layout"),
                entries: &[
                    bind_buffer(0, true),
                    bind_buffer(1, true),
                    bind_buffer(2, true),
                    bind_buffer(3, false),
                ],
            },
        );
        let module = ctx.shader("voice/gpu dwconv shader", DWCONV_WGSL)?;
        let pipeline = ctx.pipeline("voice/gpu dwconv", &module, &layout, "main");
        let mk = |label: &str, usages| ctx.create_buffer(label, 4, usages);
        Ok(Self {
            ctx: ctx.clone(),
            pipeline,
            layout,
            params: ctx.create_buffer(
                "voice/gpu dwconv params",
                16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            x_buf: mk("voice/gpu dwconv x", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            w_buf: mk("voice/gpu dwconv w", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            out_buf: mk(
                "voice/gpu dwconv out",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        })
    }

    /// `out[tt, c] = sum_k x[tt - pad_left + k, c] * w[c, k]` over valid frames.
    pub fn forward(
        &mut self,
        x: &[f32],
        w: &[f32],
        t: usize,
        d: usize,
        kh: usize,
        pad_left: usize,
    ) -> Vec<f32> {
        let count = t * d;
        let size = (count * 4).max(4) as u64;
        let wsize = (w.len() * 4).max(4) as u64;
        grow_buffer(
            &self.ctx,
            &mut self.x_buf,
            "voice/gpu dwconv x",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.w_buf,
            "voice/gpu dwconv w",
            wsize,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow_buffer(
            &self.ctx,
            &mut self.out_buf,
            "voice/gpu dwconv out",
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params: [u32; 4] = [t as u32, d as u32, kh as u32, pad_left as u32];
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voice/gpu dwconv"),
            });
        self.ctx.queue.write_buffer(&self.params, 0, &bytes32(&params));
        self.ctx.queue.write_buffer(&self.x_buf, 0, bytemuck_safe(x));
        self.ctx.queue.write_buffer(&self.w_buf, 0, bytemuck_safe(w));
        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voice/gpu dwconv bind"),
            layout: &self.layout,
            entries: &[
                binding(0, &self.params),
                binding(1, &self.x_buf),
                binding(2, &self.w_buf),
                binding(3, &self.out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voice/gpu dwconv pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(256) as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        let bytes = self.ctx.download(&self.out_buf, size);
        bytes_to_f32(&bytes, count)
    }
}

fn bind_buffer(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }),
    }
}

fn bytes32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytemuck_safe(x: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding bits; the slice is exactly len*4 bytes.
    unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    }
}

pub(crate) fn f32_bytes(x: &[f32]) -> Vec<u8> {
    bytemuck_safe(x).to_vec()
}

fn bytes_to_f32(v: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(f32::from_le_bytes([
            v[i * 4],
            v[i * 4 + 1],
            v[i * 4 + 2],
            v[i * 4 + 3],
        ]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    #[test]
    fn q8_gemm_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let mut rnd_u16 = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 48) as u16
        };

        let (t, k, n) = (13usize, 1024usize, 37usize);
        let kb = k.div_ceil(32);
        let block_bytes = 34;
        let padded_row = kb * block_bytes;
        // Raw weights: rows=n, each row kb blocks of [f16 scale][32 i8].
        let mut raw = vec![0u8; n * padded_row];
        for r in 0..n {
            for b in 0..kb {
                let base = r * padded_row + b * block_bytes;
                raw[base..base + 2].copy_from_slice(&rnd_u16().to_le_bytes());
                for i in 0..32 {
                    raw[base + 2 + i] = (rnd_f() * 127.0) as i8 as u8;
                }
            }
        }
        let mut x = vec![0.0f32; t * k];
        for v in &mut x {
            *v = rnd_f();
        }

        let packed = pack_q8(&raw, n, k, padded_row, block_bytes);
        assert_eq!(packed.q.len(), n * kb * 8 * 4);
        assert_eq!(packed.s.len(), n * kb * 4);

        let mut y_ref = vec![0.0f32; t * n];
        q8_gemm_ref(&packed, &x, t, k, &mut y_ref);

        let mut gemm = Q8Gemm::new(&ctx).unwrap();
        let w_q = ctx.upload("wq", &packed.q, wgpu::BufferUsages::STORAGE);
        let w_s = ctx.upload("ws", &packed.s, wgpu::BufferUsages::STORAGE);
        let y_gpu = gemm.gemm(&packed, &w_q, &w_s, None, &x, t, k);

        let mut max_err = 0.0f32;
        for i in 0..y_ref.len() {
            max_err = max_err.max((y_ref[i] - y_gpu[i]).abs());
        }
        let scale = y_ref
            .iter()
            .fold(0.0f32, |a, &b| a.max(b.abs()))
            .max(1.0);
        assert!(
            max_err < scale * 1e-4,
            "max_err={max_err} scale={scale} ref={:?} gpu={:?}",
            &y_ref[..8.min(y_ref.len())],
            &y_gpu[..8.min(y_gpu.len())]
        );
    }

    fn layernorm_ref(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        t: usize,
        d: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; t * d];
        for row in 0..t {
            let base = row * d;
            let mean = x[base..base + d].iter().sum::<f32>() / d as f32;
            let var = x[base..base + d]
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f32>()
                / d as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                out[base + i] =
                    (x[base + i] - mean) * rstd * gamma[i] + beta[i];
            }
        }
        out
    }

    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    #[test]
    fn layernorm_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let (t, d) = (13usize, 1024usize);
        let eps = 1e-5;
        let mut x = vec![0.0f32; t * d];
        let mut gamma = vec![0.0f32; d];
        let mut beta = vec![0.0f32; d];
        for v in &mut x {
            *v = rnd_f() * 3.0 + 1.0;
        }
        for v in &mut gamma {
            *v = 1.0 + rnd_f() * 0.5;
        }
        for v in &mut beta {
            *v = rnd_f() * 0.2;
        }
        let y_ref = layernorm_ref(&x, &gamma, &beta, t, d, eps);
        let mut ln = LayerNormKernel::new(&ctx).unwrap();
        let y_gpu = ln.forward(&x, &gamma, &beta, t, d, eps);
        let mut max_err = 0.0f32;
        for i in 0..y_ref.len() {
            max_err = max_err.max((y_ref[i] - y_gpu[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "max_err={max_err} ref={:?} gpu={:?}",
            &y_ref[..8],
            &y_gpu[..8]
        );
    }

    #[test]
    fn elementwise_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0xdead_beef_cafe_f00du64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let n = 4096;
        let mut x = vec![0.0f32; n];
        for v in &mut x {
            *v = rnd_f();
        }
        let mut ew = ElementwiseKernel::new(&ctx).unwrap();

        // silu
        let y_ref: Vec<f32> = x.iter().map(|&v| v * sigmoid(v)).collect();
        let y_gpu = ew.silu(&x);
        let mut max_err = y_ref
            .iter()
            .zip(&y_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "silu max_err={max_err}");

        // relu
        let y_ref: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
        let y_gpu = ew.relu(&x);
        max_err = y_ref
            .iter()
            .zip(&y_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "relu max_err={max_err}");

        // glu: [t, 2d] with gate block then value block
        let dim = 1024;
        let (tt_rows, dd_in) = (8usize, 2048usize);
        let mut g = vec![0.0f32; tt_rows * dd_in];
        for v in &mut g {
            *v = rnd_f();
        }
        let y_ref: Vec<f32> = (0..tt_rows * dim)
            .map(|i| {
                let row = i / dim;
                let off = i % dim;
                g[row * 2 * dim + off] * sigmoid(g[row * 2 * dim + dim + off])
            })
            .collect();
        let y_gpu = ew.glu(&g, dim);
        max_err = y_ref
            .iter()
            .zip(&y_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "glu max_err={max_err}");

        // add_mul
        let mut b = vec![0.0f32; n];
        for v in &mut b {
            *v = rnd_f();
        }
        let scale = 0.5f32;
        let y_ref: Vec<f32> =
            x.iter().zip(&b).map(|(a, bb)| a + scale * bb).collect();
        let y_gpu = ew.add_mul(&x, &b, scale);
        max_err = y_ref
            .iter()
            .zip(&y_gpu)
            .map(|(a, bb)| (a - bb).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "add_mul max_err={max_err}");

        // bias_add
        let dim = 1024;
        let bias: Vec<f32> = (0..dim).map(|_| rnd_f()).collect();
        let y_ref: Vec<f32> =
            (0..n).map(|i| x[i] + bias[i % dim]).collect();
        let y_gpu = ew.bias_add(&x, &bias);
        max_err = y_ref
            .iter()
            .zip(&y_gpu)
            .map(|(a, bb)| (a - bb).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "bias_add max_err={max_err}");
    }

    #[test]
    fn attention_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0xcafe_babe_1234_5678u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let (t, d, n_heads) = (13usize, 1024usize, 8usize);
        let (left, right) = (8usize, 3usize);
        let chunk_size = right + 1;
        let left_chunks = left / chunk_size;
        let band = |qq: usize| {
            let q_chunk = qq / chunk_size;
            let k_min = q_chunk.saturating_sub(left_chunks) * chunk_size;
            let k_max = ((q_chunk + 1) * chunk_size).min(t);
            (k_min, k_max)
        };
        let head_dim = d / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut mk = |n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            for x in &mut v {
                *x = rnd_f();
            }
            v
        };
        let q = mk(t * d);
        let k = mk(t * d);
        let v = mk(t * d);
        let q_bias = mk(d);
        let v_bias = mk(d);
        let pos = mk((2 * t - 1) * d);

        // CPU reference replicating block_forward's attention.
        let mut y_ref = vec![0.0f32; t * d];
        for qq in 0..t {
            let (k_min, k_max) = band(qq);
            for h in 0..n_heads {
                let hoff = h * head_dim;
                let mut maxv = f32::NEG_INFINITY;
                let mut sum = 0.0;
                let mut exps = vec![0.0f32; t];
                for kk in k_min..k_max {
                    let mut acc = 0.0;
                    for i in 0..head_dim {
                        let qui = q[qq * d + hoff + i] + q_bias[hoff + i];
                        let qvi = q[qq * d + hoff + i] + v_bias[hoff + i];
                        acc += qui * k[kk * d + hoff + i]
                            + qvi * pos[(kk + t - qq - 1) * d + hoff + i];
                    }
                    let s = acc * scale;
                    maxv = maxv.max(s);
                    exps[kk] = s;
                }
                for e in exps.iter_mut().take(k_max).skip(k_min) {
                    *e = (*e - maxv).exp();
                    sum += *e;
                }
                let inv = 1.0 / sum;
                for i in 0..head_dim {
                    let mut acc = 0.0;
                    for kk in k_min..k_max {
                        acc += exps[kk] * inv * v[kk * d + hoff + i];
                    }
                    y_ref[qq * d + hoff + i] = acc;
                }
            }
        }

        let mut attn = AttentionKernel::new(&ctx).unwrap();
        let y_gpu =
            attn.forward(&q, &k, &v, &q_bias, &v_bias, &pos, t, d, n_heads, scale, left, right);
        let mut max_err = 0.0f32;
        for i in 0..y_ref.len() {
            max_err = max_err.max((y_ref[i] - y_gpu[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "max_err={max_err} ref={:?} gpu={:?}",
            &y_ref[..8],
            &y_gpu[..8]
        );
    }

    #[test]
    fn attn_stream_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0x5150_1503_7265_616du64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let (d, n_heads) = (1024usize, 8usize);
        let head_dim = d / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Two batches: first batch c=4 at s=0 (band = [0,4)), then a second
        // batch c=4 at s=4 where the band reuses cached k/v plus new frames.
        let chunk = 4usize;
        let left_chunks = 2usize;
        let pos_off = 3usize;
        // k_lo = saturating_sub(s/chunk - left_chunks, 0) * chunk = 8.
        let (s, c, k_lo, k_hi) = (16usize, 8usize, 8usize, 24usize);
        let band = k_hi - k_lo;
        let n_pos = 2 * chunk * left_chunks + chunk - 1 + 1; // >= 63-ish
        let mut mk = |n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            for x in &mut v {
                *x = rnd_f();
            }
            v
        };
        let q = mk(c * d);
        let kv = mk(band * d);
        let vv = mk(band * d);
        let pos_p = mk(n_pos * d);
        let q_bias = mk(d);
        let v_bias = mk(d);

        // CPU reference replicating StreamingEncoder block_new's attention.
        let mut y_ref = vec![0.0f32; c * d];
        for qi in 0..c {
            let qq = s + qi;
            let q_chunk = qq / chunk;
            let k_min = q_chunk.saturating_sub(left_chunks) * chunk;
            let k_max = ((q_chunk + 1) * chunk).min(k_hi);
            let k0 = k_min - k_lo;
            let k1 = k_max - k_lo;
            for h in 0..n_heads {
                let hoff = h * head_dim;
                let mut maxv = f32::NEG_INFINITY;
                let mut sum = 0.0;
                let mut exps = vec![0.0f32; k1];
                for kk in k0..k1 {
                    let fr = k_lo + kk;
                    let pr = qq as isize - fr as isize + pos_off as isize;
                    let mut acc = 0.0;
                    for i in 0..head_dim {
                        let qui = q[qi * d + hoff + i] + q_bias[hoff + i];
                        let qvi = q[qi * d + hoff + i] + v_bias[hoff + i];
                        acc += qui * kv[kk * d + hoff + i]
                            + qvi * pos_p[pr as usize * d + hoff + i];
                    }
                    let s = acc * scale;
                    maxv = maxv.max(s);
                    exps[kk] = s;
                }
                for e in exps.iter_mut().take(k1).skip(k0) {
                    *e = (*e - maxv).exp();
                    sum += *e;
                }
                let inv = 1.0 / sum;
                for i in 0..head_dim {
                    let mut acc = 0.0;
                    for kk in k0..k1 {
                        acc += exps[kk] * inv * vv[kk * d + hoff + i];
                    }
                    y_ref[qi * d + hoff + i] = acc;
                }
            }
        }

        let mut attn = AttnStreamKernel::new(&ctx).unwrap();
        let y_gpu = attn.forward(
            &q,
            &kv,
            &vv,
            &pos_p,
            &q_bias,
            &v_bias,
            c,
            d,
            n_heads,
            scale,
            s,
            k_lo,
            band,
            chunk,
            left_chunks,
            k_hi,
            pos_off,
        );
        let mut max_err = 0.0f32;
        for i in 0..y_ref.len() {
            max_err = max_err.max((y_ref[i] - y_gpu[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "max_err={max_err} ref={:?} gpu={:?}",
            &y_ref[..8],
            &y_gpu[..8]
        );
    }

    #[test]
    fn dwconv_matches_reference() {
        let ctx = GpuContext::init().expect("no GPU adapter");
        let ctx = Arc::new(ctx);
        let mut seed = 0xf00d_baad_9876_5432u64;
        let mut rnd_f = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let (t, d, kh, pad_left) = (13usize, 1024usize, 9usize, 8usize);
        let x: Vec<f32> = (0..t * d).map(|_| rnd_f()).collect();
        let w: Vec<f32> = (0..d * kh).map(|_| rnd_f()).collect();
        let mut y_ref = vec![0.0f32; t * d];
        for tt in 0..t {
            for dd in 0..d {
                let mut acc = 0.0;
                let t0 = tt as isize - pad_left as isize;
                for k in 0..kh {
                    let ti = t0 + k as isize;
                    if ti < 0 || ti as usize >= t {
                        continue;
                    }
                    acc += x[ti as usize * d + dd] * w[dd * kh + k];
                }
                y_ref[tt * d + dd] = acc;
            }
        }
        let mut dw = DwConvKernel::new(&ctx).unwrap();
        let y_gpu = dw.forward(&x, &w, t, d, kh, pad_left);
        let mut max_err = 0.0f32;
        for i in 0..y_ref.len() {
            max_err = max_err.max((y_ref[i] - y_gpu[i]).abs());
        }
        assert!(
            max_err < 1e-4,
            "max_err={max_err} ref={:?} gpu={:?}",
            &y_ref[..8],
            &y_gpu[..8]
        );
    }
}
