//! Batched GPU compute: many kernel dispatches recorded into a single
//! command encoder and submitted once.
//!
//! The kernel `*_record` methods in [`super::kernels`] write their params
//! and record a compute pass into a [`ComputeBatch`] instead of submitting
//! and downloading per op, so intermediate activations stay on the GPU.
//! wgpu inserts buffer transitions (write->read) automatically between
//! passes; only the small persistent results are downloaded after the
//! submit.
//!
//! One compute pass stays open for the whole batch: per-dispatch pass
//! boundaries are expensive on weak GPUs (an Intel iGPU spends ~100 us
//! per pass), and a batch records hundreds of dispatches. The pass is
//! closed around `copy` (which needs the raw encoder) and on `submit`.
//!
//! Scratch slots are bumped-allocated per batch. A slot is written by
//! exactly one op per batch (either host-side via `write` before the
//! submission, or by one compute pass) and read by later passes, so the
//! immediate `queue.write_buffer` calls never race with recorded passes.

use super::context::GpuContext;

/// Minimum scratch slot allocation (bytes). The largest single activation
/// in the streaming block is `[band, d]` (64 x 1024 f32 = 256 KiB), but
/// slots are aligned modestly because a full batch (all 24 blocks in one
/// submission) bumps ~700 slots; 64 KiB keeps the pool at ~50 MiB.
const SLOT_ALIGN: u64 = 64 << 10;

/// One reusable submission: owns the command encoder and a pool of
/// storage buffers reused across ops (and across blocks via `reset`).
///
/// Safety of the `'static` compute pass: `pass` borrows `encoder`, hidden
/// behind a lifetime cast. Invariants (enforced here, no caller
/// cooperation needed):
/// - the pass is always dropped (`take`) before `encoder` is replaced in
///   `submit()` or the batch is dropped (`Drop`);
/// - `copy()` drops the pass before touching `encoder`, so wgpu never
///   sees encoder use while a pass is open.
#[derive(Debug)]
pub struct ComputeBatch<'a> {
    pub ctx: &'a GpuContext,
    encoder: wgpu::CommandEncoder,
    #[allow(unsafe_code)]
    pass: Option<wgpu::ComputePass<'static>>,
    scratch: Vec<Option<wgpu::Buffer>>,
    next_scratch: usize,
}

impl<'a> ComputeBatch<'a> {
    pub fn new(ctx: &'a GpuContext) -> Self {
        Self {
            ctx,
            encoder: new_encoder(ctx),
            pass: None,
            scratch: Vec::new(),
            next_scratch: 0,
        }
    }

    /// Close the open compute pass (if any) - the only operation that
    /// may touch `self.encoder` outside `dispatch`.
    fn end_pass(&mut self) {
        self.pass.take();
    }

    /// Bump-allocate a scratch slot of at least `size` bytes and return
    /// its handle (owned clone). Slots are reused across batches.
    pub fn alloc(&mut self, size: u64) -> wgpu::Buffer {
        let idx = self.next_scratch;
        self.next_scratch += 1;
        self.buf(idx, size)
    }

    /// Scratch slot `idx`, returned as an owned handle. Grows the slot to
    /// `size` bytes when it is too small (only ever happens across batches,
    /// after the previous submission's results are downloaded).
    pub fn buf(&mut self, idx: usize, size: u64) -> wgpu::Buffer {
        let aligned = size.div_ceil(SLOT_ALIGN) * SLOT_ALIGN;
        if self.scratch.len() <= idx {
            self.scratch.resize(idx + 1, None);
        }
        match self.scratch[idx].as_mut() {
            Some(slot) if slot.size() >= aligned => slot.clone(),
            _ => {
                let slot = self.ctx.create_buffer(
                    "voice/gpu batch scratch",
                    aligned,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                );
                self.scratch[idx] = Some(slot.clone());
                slot
            }
        }
    }

    /// Start a fresh command buffer for the next batch, keeping the
    /// scratch pool. Must only be called after the previous submission's
    /// results have been downloaded.
    pub fn reset(&mut self) {
        self.end_pass();
        self.next_scratch = 0;
    }

    /// Upload host bytes into `buf` (executed before the submission, in
    /// call order). `buf` must have `COPY_DST`.
    pub fn write(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.ctx.queue.write_buffer(buf, 0, data);
    }

    /// GPU-to-GPU copy within the current submission.
    pub fn copy(
        &mut self,
        src: &wgpu::Buffer,
        src_off: u64,
        dst: &wgpu::Buffer,
        dst_off: u64,
        size: u64,
    ) {
        self.end_pass();
        self.encoder
            .copy_buffer_to_buffer(src, src_off, dst, dst_off, size);
    }

    /// Record one compute dispatch into the batch's single open pass.
    #[allow(unsafe_code)]
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        wg_x: u32,
        wg_y: u32,
    ) {
        if self.pass.is_none() {
            let pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("voice/gpu batch pass"),
                    timestamp_writes: None,
                });
            // SAFETY: the pass borrows `self.encoder`; it is always
            // dropped before `encoder` is replaced or the batch drops
            // (see the struct-level invariants), and wgpu sees the
            // encoder as free whenever `copy`/`submit` touch it.
            self.pass =
                Some(unsafe {
                    std::mem::transmute::<
                        wgpu::ComputePass<'_>,
                        wgpu::ComputePass<'static>,
                    >(pass)
                });
        }
        // SAFETY: pass is Some here; set_pipeline/set_bind_group/dispatch
        // are exactly what the pass API allows in sequence.
        let pass = unsafe { self.pass.as_mut().unwrap_unchecked() };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Submit the pending command buffer and start a fresh one, keeping
    /// the scratch pool for the next batch.
    pub fn submit(&mut self) {
        self.end_pass();
        let cmd = std::mem::replace(&mut self.encoder, new_encoder(self.ctx));
        self.ctx.queue.submit(Some(cmd.finish()));
        self.next_scratch = 0;
    }
}

impl Drop for ComputeBatch<'_> {
    fn drop(&mut self) {
        self.end_pass();
    }
}

fn new_encoder(ctx: &GpuContext) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voice/gpu batch"),
        })
}
