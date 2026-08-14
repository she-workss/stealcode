//! Model hyperparameters parsed from the GGUF metadata KV table.

use anyhow::{Context, Result, bail};

use crate::gguf::{Gguf, GgufValue};

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub preemph: f32,
    pub dither: f32,
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub feat_in: usize,
    pub subsampling_factor: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    /// causal -> (kernel-1, 0)
    pub conv_context_left: usize,
    pub conv_context_right: usize,
    pub att_context_left: usize,
    pub att_context_right: usize,
    pub xscaling: bool,
    pub use_bias: bool,
    pub pos_emb_max_len: usize,
    /// Multilingual prompt-conditioned variants: one-hot width of the
    /// prompt MLP (concatenated after the encoder output).
    pub num_prompts: usize,
}

#[derive(Debug, Clone)]
pub struct RnntConfig {
    pub vocab_size: usize,
    pub blank_id: usize,
    pub pred_embed_dim: usize,
    pub pred_hidden: usize,
    pub pred_n_layers: usize,
    pub joint_dim: usize,
    pub num_prompts: usize,
    pub prompt_intermediate: usize,
    pub max_symbols_per_step: usize,
    pub prompt_dictionary: Vec<(String, u32)>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub preprocessor: PreprocessorConfig,
    pub encoder: EncoderConfig,
    pub rnnt: RnntConfig,
    pub tokens: Vec<String>,
}

fn kv_u32(gguf: &Gguf, key: &str) -> Result<u32> {
    gguf.kv
        .get(key)
        .and_then(GgufValue::as_u32)
        .with_context(|| format!("GGUF metadata: missing u32 {key}"))
}

fn kv_bool(gguf: &Gguf, key: &str) -> Result<bool> {
    match gguf.kv.get(key) {
        Some(GgufValue::U8(v)) => Ok(*v != 0),
        Some(GgufValue::U32(v)) => Ok(*v != 0),
        _ => bail!("GGUF metadata: missing bool {key}"),
    }
}

fn kv_str<'a>(gguf: &'a Gguf, key: &str) -> Result<&'a str> {
    gguf.kv
        .get(key)
        .and_then(GgufValue::as_str)
        .with_context(|| format!("GGUF metadata: missing string {key}"))
}

fn kv_f32(gguf: &Gguf, key: &str) -> Result<f32> {
    gguf.kv
        .get(key)
        .and_then(GgufValue::as_f32)
        .with_context(|| format!("GGUF metadata: missing f32 {key}"))
}

impl ModelConfig {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let sample_rate =
            kv_u32(gguf, "asr.preprocessor.sample_rate")? as usize;
        let n_fft = kv_u32(gguf, "asr.preprocessor.n_fft")? as usize;
        let window_secs = kv_f32(gguf, "asr.preprocessor.window_size")?;
        let stride_secs = kv_f32(gguf, "asr.preprocessor.window_stride")?;
        let win_length = (window_secs * sample_rate as f32).round() as usize;
        let hop = (stride_secs * sample_rate as f32).round() as usize;
        let n_mels = kv_u32(gguf, "asr.preprocessor.features")? as usize;

        let conv_kernel =
            kv_u32(gguf, "asr.encoder.conv_kernel_size")? as usize;
        let conv_context = kv_str(gguf, "asr.encoder.conv_context")?;
        let (conv_context_left, conv_context_right) = match conv_context {
            "causal" => (conv_kernel - 1, 0),
            "same" => ((conv_kernel - 1) / 2, (conv_kernel - 1) / 2),
            other => bail!("GGUF metadata: unsupported conv_context {other}"),
        };

        let prompt_dictionary = match gguf.kv.get("asr.rnnt.prompt_dictionary")
        {
            Some(GgufValue::ArrStr(v)) => v
                .iter()
                .map(|e| {
                    let (loc, id) = e.rsplit_once(':').with_context(|| {
                        format!("bad prompt_dictionary entry {e}")
                    })?;
                    Ok((loc.to_string(), id.parse().context("bad prompt id")?))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!("GGUF metadata: missing prompt_dictionary"),
        };

        let tokens = match gguf.kv.get("asr.tokenizer.vocab") {
            Some(GgufValue::ArrStr(v)) => v.clone(),
            _ => bail!("GGUF metadata: missing tokenizer vocab"),
        };

        Ok(Self {
            preprocessor: PreprocessorConfig {
                sample_rate,
                n_fft,
                win_length,
                hop,
                n_mels,
                preemph: kv_f32(gguf, "asr.preprocessor.preemph")?,
                dither: kv_f32(gguf, "asr.preprocessor.dither")?,
            },
            encoder: EncoderConfig {
                d_model: kv_u32(gguf, "asr.encoder.d_model")? as usize,
                n_layers: kv_u32(gguf, "asr.encoder.n_layers")? as usize,
                n_heads: kv_u32(gguf, "asr.encoder.n_heads")? as usize,
                d_ff: kv_u32(gguf, "asr.encoder.d_ff")? as usize,
                feat_in: kv_u32(gguf, "asr.encoder.feat_in")? as usize,
                subsampling_factor: kv_u32(
                    gguf,
                    "asr.encoder.subsampling_factor",
                )? as usize,
                conv_channels: kv_u32(
                    gguf,
                    "asr.encoder.subsampling_conv_channels",
                )? as usize,
                conv_kernel,
                conv_context_left,
                conv_context_right,
                att_context_left: kv_u32(gguf, "asr.encoder.offline_left_ctx")?
                    as usize,
                // NVIDIA streaming config: [left, right] in 80ms frames,
                // chunk size = right + 1. GGUF ships offline_right_ctx=3
                // (chunk 4 = 0.32s); we raise it to 6 (chunk 7 = 0.56s)
                // for a better accuracy/latency trade-off.
                att_context_right: 6,
                xscaling: kv_bool(gguf, "asr.encoder.xscaling")?,
                use_bias: kv_bool(gguf, "asr.encoder.use_bias")?,
                pos_emb_max_len: kv_u32(gguf, "asr.encoder.pos_emb_max_len")?
                    as usize,
                num_prompts: kv_u32(gguf, "asr.rnnt.num_prompts")? as usize,
            },
            rnnt: RnntConfig {
                vocab_size: kv_u32(gguf, "asr.rnnt.vocab_size")? as usize,
                blank_id: kv_u32(gguf, "asr.rnnt.blank_id")? as usize,
                pred_embed_dim: kv_u32(gguf, "asr.rnnt.pred_embed_dim")?
                    as usize,
                pred_hidden: kv_u32(gguf, "asr.rnnt.pred_hidden")? as usize,
                pred_n_layers: kv_u32(gguf, "asr.rnnt.pred_num_layers")?
                    as usize,
                joint_dim: kv_u32(gguf, "asr.rnnt.joint_dim")? as usize,
                num_prompts: kv_u32(gguf, "asr.rnnt.num_prompts")? as usize,
                prompt_intermediate: 2048,
                max_symbols_per_step: kv_u32(
                    gguf,
                    "asr.rnnt.max_symbols_per_step",
                )? as usize,
                prompt_dictionary,
            },
            tokens,
        })
    }

    /// Resolve a language hint to a prompt index (transcribe.cpp
    /// resolve_prompt_id): exact match in the dictionary; empty hint ->
    /// the dictionary's auto slot ("auto" entry); unknown -> error.
    pub fn resolve_prompt_id(&self, language: Option<&str>) -> Result<u32> {
        match language {
            Some(lang) if !lang.is_empty() => self
                .rnnt
                .prompt_dictionary
                .iter()
                .find(|(loc, _)| loc == lang)
                .map(|(_, id)| *id)
                .with_context(|| {
                    format!("language {lang} not in prompt dictionary")
                }),
            _ => self
                .rnnt
                .prompt_dictionary
                .iter()
                .find(|(loc, _)| loc == "auto")
                .map(|(_, id)| *id)
                .or_else(|| {
                    self.rnnt.prompt_dictionary.first().map(|(_, id)| *id)
                })
                .context("empty prompt dictionary"),
        }
    }
}
