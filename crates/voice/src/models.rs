//! Model provisioning: auto-downloads the Nemotron GGUF on first use.
//!
//! The checkpoint lands in `paths::model_dir()`; no env vars or manual
//! setup are needed.

use std::{
    fs::File,
    io::{Read, Write},
};

use anyhow::{Context, Result};
use audio_engine::Nemotron;
use tracing::info;

/// Load the model, downloading the checkpoint from the Hub on first
/// use. `on_progress(done, total)` reports download progress (`total`
/// is `None` when unknown).
pub fn load_model(
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<Nemotron> {
    let path = ensure_model_available(&mut on_progress)?;
    Nemotron::load(&path)
}

/// Ensure the GGUF exists locally, downloading it when missing.
fn ensure_model_available(
    on_progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<std::path::PathBuf> {
    let path = paths::model_path();
    if path.exists() {
        return Ok(path.clone());
    }
    let dir = paths::model_dir();
    std::fs::create_dir_all(dir)?;
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        paths::MODEL_REPO,
        paths::GGUF_FILE
    );
    info!("downloading {url}");
    let mut resp = reqwest::blocking::get(&url)
        .with_context(|| format!("download {url}"))?
        .error_for_status()?;
    let total = resp.content_length();
    let tmp = dir.join(format!("{}.part", paths::GGUF_FILE));
    let mut out = File::create(&tmp)?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    loop {
        let n = resp.read(&mut buf).context("read response body")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        on_progress(done, total);
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp, path).with_context(|| {
        format!("move {} -> {}", tmp.display(), path.display())
    })?;
    Ok(path.clone())
}
