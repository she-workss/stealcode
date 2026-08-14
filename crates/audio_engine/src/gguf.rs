//! Minimal GGUF loader for the nemotron-3.5-asr-streaming-0.6b Q8_0
//! checkpoint (NVIDIA-format GGUF produced by gguf-py).
//!
//! Layout notes verified empirically against the file:
//!   * header `<IIQQ`: magic, version, n_tensors, n_kv
//!   * KV value type table (as written by this converter): t==2 | t==8  ->
//!     string (u64 len + bytes) t==4        -> u32 t==5        -> i32 t==6 ->
//!     f32 t==7        -> u8 (bool) t==3 | t==9 -> array (et u32, count u64,
//!     elements) array element types: 4/5 -> u32/i32, 6 -> f32, 8 -> string
//!   * tensor table: name str, n_dims u32, dims u64[n], dtype u32, offset u64
//!     (dtype AFTER dims - non-standard, but what the file contains)
//!   * tensor data starts at the table end rounded up to 32 bytes; per-tensor
//!     offsets are relative to that start. Dtypes seen in this file: 0=f32,
//!     1=f16, 7=q8_0, 8=q8_1.
//!
//! Tensor element order in memory equals torch row-major of the
//! reversed gguf dims (the gguf-py convention), so `dims` as stored
//! directly give: `dims[0]` = fastest axis.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

const GGUF_MAGIC: u32 = 0x4655_4747;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U8(u8),
    Str(String),
    ArrU32(Vec<u32>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrStr(Vec<String>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(*v),
            Self::U32(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::U8(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_arr_u32(&self) -> Option<&[u32]> {
        match self {
            Self::ArrU32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_arr_str(&self) -> Option<&[String]> {
        match self {
            Self::ArrStr(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    /// gguf dims, fastest axis first (== reversed torch shape).
    pub dims: Vec<u64>,
    pub dtype: u32,
    /// Byte offset of the tensor data, relative to the data start.
    pub offset: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub kv: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorMeta>,
    bytes: Mmap,
    data_start: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8> {
        let v = *self.bytes.get(self.pos).context("gguf: unexpected EOF")?;
        self.pos += 1;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 4)
            .context("gguf: unexpected EOF")?;
        self.pos += 4;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 8)
            .context("gguf: unexpected EOF")?;
        self.pos += 8;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let s = self
            .bytes
            .get(self.pos..self.pos + len)
            .context("gguf: string out of range")?;
        self.pos += len;
        Ok(String::from_utf8_lossy(s).into_owned())
    }

    fn array(&mut self) -> Result<GgufValue> {
        let et = self.u32()?;
        let count = self.u64()? as usize;
        match et {
            4 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.u32()?);
                }
                Ok(GgufValue::ArrU32(v))
            }
            5 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.i32()?);
                }
                Ok(GgufValue::ArrI32(v))
            }
            6 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.f32()?);
                }
                Ok(GgufValue::ArrF32(v))
            }
            8 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push(self.string()?);
                }
                Ok(GgufValue::ArrStr(v))
            }
            other => bail!("gguf: unsupported array element type {other}"),
        }
    }
}

impl Gguf {
    /// Memory-map the file (read-only); tensor data is sliced out of the
    /// mapping lazily, so loading costs no extra 1.4 GB heap copy and the
    /// peak RAM stays near the weights' own footprint.
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open GGUF {}", path.display()))?;
        // SAFETY: the mapping is read-only and no other code mutates the
        // file while it is open.
        let bytes = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap GGUF {}", path.display()))?;
        let (kv, tensors, data_start) = Self::parse(&bytes)?;
        Ok(Self {
            kv,
            tensors,
            bytes,
            data_start,
        })
    }

    fn parse(
        slice: &[u8],
    ) -> Result<(HashMap<String, GgufValue>, Vec<TensorMeta>, usize)> {
        let mut r = Reader {
            bytes: slice,
            pos: 0,
        };
        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (bad magic)");
        }
        let _version = r.u32()?;
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut kv = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let t = r.u32()?;
            let value = match t {
                2 | 8 => GgufValue::Str(r.string()?),
                4 => GgufValue::U32(r.u32()?),
                5 => GgufValue::I32(r.i32()?),
                6 => GgufValue::F32(r.f32()?),
                7 => GgufValue::U8(r.u8()?),
                3 | 9 => r.array()?,
                other => bail!(
                    "gguf: unsupported KV value type {other} for key {key}"
                ),
            };
            kv.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let dtype = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorMeta {
                name,
                dims,
                dtype,
                offset,
            });
        }

        let data_start = (r.pos + 31) & !31;
        Ok((kv, tensors, data_start))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor (quantized layout as stored).
    pub fn tensor_data(&self, meta: &TensorMeta) -> Result<&[u8]> {
        let base = self.data_start + meta.offset as usize;
        let end = base
            .checked_add(self.tensor_bytes(meta)?)
            .context("gguf: tensor offset overflow")?;
        self.bytes
            .get(base..end)
            .context("gguf: tensor data out of range")
    }

    /// Dequantize a tensor into f32, in the stored (torch row-major)
    /// element order.
    pub fn read_f32(&self, meta: &TensorMeta) -> Result<Vec<f32>> {
        let n = meta
            .dims
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(*d))
            .context("gguf: tensor too large")? as usize;
        let data = self.tensor_data(meta)?;
        let mut out = Vec::with_capacity(n);
        match meta.dtype {
            0 => {
                for chunk in data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]));
                }
            }
            1 => {
                for chunk in data.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([
                        chunk[0], chunk[1],
                    ])));
                }
            }
            7 => dequant_q8_0(data, meta.dims[0] as usize, &mut out)?,
            8 => dequant_q8f16(data, meta.dims[0] as usize, &mut out)?,
            other => bail!(
                "gguf: unsupported tensor dtype {other} for {}",
                meta.name
            ),
        }
        Ok(out)
    }

    fn tensor_bytes(&self, meta: &TensorMeta) -> Result<usize> {
        let n = meta
            .dims
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(*d))
            .context("gguf: tensor too large")? as usize;
        let row = meta.dims.first().copied().unwrap_or(1) as usize;
        match meta.dtype {
            0 => Ok(n * 4),
            1 => Ok(n * 2),
            7 => {
                let blocks = (n / row) * row.div_ceil(32);
                Ok(blocks * 36)
            }
            8 => {
                let blocks = (n / row) * row.div_ceil(32);
                Ok(blocks * 34)
            }
            other => bail!(
                "gguf: unsupported tensor dtype {other} for {}",
                meta.name
            ),
        }
    }
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if man == 0 {
                sign << 31
            } else {
                // subnormal
                let e = 127 - 15 + 1;
                let m = man << 13;
                let mut e = e;
                let mut m = m;
                while m & 0x0080_0000 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x007f_ffff;
                (sign << 31) | ((e as u32) << 23) | m
            }
        }
        0x1f => {
            if man == 0 {
                (sign << 31) | 0x7f80_0000
            } else {
                (sign << 31) | 0x7fc0_0000 | (man << 13)
            }
        }
        e => (sign << 31) | ((e + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Q8_0: blocks of 32 values, each block = f32 scale + 32 i8.
/// Each row is padded to a multiple of 32 elements.
fn dequant_q8_0(data: &[u8], row: usize, out: &mut Vec<f32>) -> Result<()> {
    let blocks_per_row = row.div_ceil(32);
    let block_bytes = 36usize;
    for (blk, base) in
        (0..data.len() / block_bytes).zip((0..data.len()).step_by(block_bytes))
    {
        let scale = f32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]);
        let vals = &data[base + 4..base + 36];
        let in_row = (blk % blocks_per_row) * 32;
        let keep = (32usize).min(row.saturating_sub(in_row));
        for j in 0..keep {
            out.push(scale * (vals[j] as i8 as f32));
        }
    }
    Ok(())
}

/// Q8F16 (this model's Q8_0): blocks of 32 values, each block =
/// f16 d + 32 i8; value = d * q. 34 bytes per block.
fn dequant_q8f16(data: &[u8], row: usize, out: &mut Vec<f32>) -> Result<()> {
    let blocks_per_row = row.div_ceil(32);
    let block_bytes = 34usize;
    for (blk, base) in
        (0..data.len() / block_bytes).zip((0..data.len()).step_by(block_bytes))
    {
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let vals = &data[base + 2..base + 34];
        let in_row = (blk % blocks_per_row) * 32;
        let keep = (32usize).min(row.saturating_sub(in_row));
        for j in 0..keep {
            out.push(d * (vals[j] as i8 as f32));
        }
    }
    Ok(())
}
