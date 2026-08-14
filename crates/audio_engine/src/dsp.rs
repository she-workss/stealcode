//! Mel-spectrogram frontend mirroring NeMo's
//! AudioToMelSpectrogramPreprocessor / FilterbankFeatures:
//!   preemphasis (0.97), STFT (n_fft, hop, hann symmetric,
//!   zero-padded `constant` center padding), power spectrum,
//!   mel filterbank (from the GGUF `preprocessor.fb` tensor),
//!   log(x + 2^-24). Dithering is training-only in NeMo, so it is
//!   not applied here.

use anyhow::{Result, bail};
use tracing::debug;

use crate::nemotron::config::PreprocessorConfig;

const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24

#[derive(Debug)]
pub struct MelFrontend {
    cfg: PreprocessorConfig,
    /// hann(win_length, symmetric) zero-padded to n_fft.
    window: Vec<f32>,
    /// Mel filterbank [n_fft/2 + 1, n_mels] (from GGUF).
    fb: Vec<f32>,
    /// Precomputed twiddle factors (radix-2 FFT, n_fft must be a
    /// power of two).
    twiddles: Vec<f32>,
    fft_re: Vec<f32>,
    fft_im: Vec<f32>,
}

impl MelFrontend {
    pub fn new(cfg: PreprocessorConfig, fb: &[f32]) -> Result<Self> {
        if !cfg.n_fft.is_power_of_two() {
            bail!("n_fft {} is not a power of two", cfg.n_fft);
        }
        let n_freq = cfg.n_fft / 2 + 1;
        let fb_len = n_freq * cfg.n_mels;
        if fb.len() < fb_len {
            bail!("mel filterbank too small: {} < {fb_len}", fb.len());
        }
        let fb = fb[..fb_len].to_vec();

        // torch.hann_window(win_length, periodic=False)
        let mut window = vec![0.0f32; cfg.n_fft];
        let wl = cfg.win_length;
        for n in 0..wl {
            let x = 2.0 * std::f32::consts::PI * n as f32 / (wl as f32 - 1.0);
            window[n] = 0.5 * (1.0 - x.cos());
        }

        let n = cfg.n_fft;
        let mut twiddles = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }

        Ok(Self {
            cfg,
            window,
            fb,
            twiddles,
            fft_re: vec![0.0; n],
            fft_im: vec![0.0; n],
        })
    }

    /// Frame count for `n_samples` (librosa/torch center=True):
    /// n // hop + 1.
    pub fn n_frames(&self, n_samples: usize) -> usize {
        n_samples / self.cfg.hop + 1
    }

    /// Compute the mel spectrogram of a full utterance.
    /// Returns frame-major [n_frames, n_mels] f32.
    pub fn compute(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let (n_fft, hop, n_mels, preemph) = {
            let c = &self.cfg;
            (c.n_fft, c.hop, c.n_mels, c.preemph)
        };
        let n = pcm.len();
        let n_frames = self.n_frames(n);
        let half = n_fft / 2;

        // Preemphasis.
        let mut sig = vec![0.0f32; n];
        if n > 0 {
            sig[0] = pcm[0];
            for i in 1..n {
                sig[i] = pcm[i] - preemph * pcm[i - 1];
            }
        }

        // Zero (constant) padding, n_fft/2 each side.
        let padded_len = n + n_fft;
        let mut padded = vec![0.0f32; padded_len];
        padded[half..half + n].copy_from_slice(&sig);
        debug!(
            "pcm[..4]={:?} pcm_max={} sig_max={}",
            &pcm[..4],
            pcm.iter().fold(0.0f32, |a, &b| a.max(b.abs())),
            sig.iter().fold(0.0f32, |a, &b| a.max(b.abs()))
        );

        let mut spec = vec![0.0f32; n_frames * (half + 1)];
        for t in 0..n_frames {
            let start = t * hop;
            self.fft_frame(&padded[start..start + n_fft]);
            let row = &mut spec[t * (half + 1)..(t + 1) * (half + 1)];
            for k in 0..=half {
                let re = self.fft_re[k];
                let im = self.fft_im[k];
                row[k] = re * re + im * im;
            }
        }

        // mel = power @ fb; log(x + 2^-24).
        let mut mel = vec![0.0f32; n_frames * n_mels];
        let n_freq = half + 1;
        debug!(
            "spec[0][..4] = {:?}, spec[0][200..204] = {:?}",
            &spec[..4],
            &spec[200..204]
        );
        debug!(
            "fb[0][..4] = {:?}, fb[100][..4] = {:?}",
            &self.fb[..4],
            &self.fb[100 * n_freq..100 * n_freq + 4]
        );
        for t in 0..n_frames {
            let spec_row = &spec[t * n_freq..(t + 1) * n_freq];
            let mel_row = &mut mel[t * n_mels..(t + 1) * n_mels];
            for m in 0..n_mels {
                let fb_col = &self.fb[m * n_freq..(m + 1) * n_freq];
                let mut acc = 0.0f64;
                for k in 0..n_freq {
                    acc += spec_row[k] as f64 * fb_col[k] as f64;
                }
                mel_row[m] = (acc as f32 + LOG_ZERO_GUARD).ln();
            }
        }
        Ok(mel)
    }

    /// STFT column for one window (in-place, no windowing by caller).
    fn fft_frame(&mut self, frame: &[f32]) {
        let n = self.cfg.n_fft;
        let fft_re = &mut self.fft_re;
        let fft_im = &mut self.fft_im;
        let window = &self.window;
        for i in 0..n {
            fft_re[i] = frame[i] * window[i];
            fft_im[i] = 0.0;
        }
        // Iterative radix-2 Cooley-Tukey with bit-reversal permutation.
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                fft_re.swap(i, j);
                fft_im.swap(i, j);
            }
        }
        let tw = &self.twiddles;
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for i in (0..n).step_by(len) {
                for k in 0..half {
                    let t_re = fft_re[i + k + half] * tw[2 * k * step]
                        - fft_im[i + k + half] * tw[2 * k * step + 1];
                    let t_im = fft_re[i + k + half] * tw[2 * k * step + 1]
                        + fft_im[i + k + half] * tw[2 * k * step];
                    fft_re[i + k + half] = fft_re[i + k] - t_re;
                    fft_im[i + k + half] = fft_im[i + k] - t_im;
                    fft_re[i + k] += t_re;
                    fft_im[i + k] += t_im;
                }
            }
            len <<= 1;
        }
    }
}

/// Incremental mel frontend for streaming: mel frames are produced
/// only once their STFT window is fully inside the received signal,
/// so they never need recomputation. The final (held-back) frames are
/// computed at end-of-stream via `finish`.
#[derive(Debug)]
pub struct StreamingMel {
    frontend: MelFrontend,
    /// Preemphasized samples received so far.
    pcm: Vec<f32>,
    /// Stable mel frames, frame-major [n, n_mels].
    mel: Vec<f32>,
    n_mels: usize,
    last_preemph_sample: f32,
}

impl StreamingMel {
    pub fn new(frontend: MelFrontend) -> Self {
        let n_mels = frontend.cfg.n_mels;
        Self {
            frontend,
            pcm: Vec::new(),
            mel: Vec::new(),
            n_mels,
            last_preemph_sample: 0.0,
        }
    }

    /// Append new PCM samples (preemphasized incrementally).
    pub fn push(&mut self, samples: &[f32]) {
        let preemph = self.frontend.cfg.preemph;
        for (i, &s) in samples.iter().enumerate() {
            let prev = if i == 0 {
                if self.pcm.is_empty() {
                    0.0
                } else {
                    self.last_preemph_sample
                }
            } else {
                samples[i - 1]
            };
            self.pcm.push(s - preemph * prev);
        }
        self.last_preemph_sample = self.pcm.last().copied().unwrap_or(0.0);
        self.grow();
    }

    fn grow(&mut self) {
        let cfg = &self.frontend.cfg;
        let n = self.pcm.len();
        let half = cfg.n_fft / 2;
        // Frame t is stable once its window end is within the signal:
        // t*hop + half <= n  ->  t <= (n - half) / hop.
        if n <= half {
            return;
        }
        let stable = (n - half) / cfg.hop + 1;
        let have = self.mel.len() / self.n_mels;
        if stable <= have {
            return;
        }
        // Recompute the last stable frame(s) using the zero-padded
        // approach (frames are immutable once computed, so just append
        // the new ones computed against the full signal).
        let Ok(mut all) = self.frontend.compute(&self.pcm) else {
            return;
        };
        let stable = stable.min(all.len() / self.n_mels);
        all.truncate(stable * self.n_mels);
        if have < stable {
            let take = stable - have;
            let start = all.len() - take * self.n_mels;
            self.mel.extend_from_slice(&all[start..]);
        }
    }

    /// Stable frames available right now.
    pub fn stable_frames(&self) -> usize {
        self.mel.len() / self.n_mels
    }

    /// Mel frame by absolute index, frame-major.
    pub fn frame(&self, t: usize) -> &[f32] {
        &self.mel[t * self.n_mels..(t + 1) * self.n_mels]
    }

    /// Total frames for the received signal (including the held-back
    /// tail that `stable_frames` excludes). Call `finish` first for the
    /// full count to be meaningful.
    pub fn total_frames(&self) -> usize {
        self.frontend.n_frames(self.pcm.len())
    }

    /// Compute the remaining tail frames (reflect-free: zero padding
    /// against the true signal end, as in the full-audio mel).
    pub fn finish(&mut self) {
        let total = self.total_frames();
        let have = self.mel.len() / self.n_mels;
        if total <= have {
            return;
        }
        let all = self.frontend.compute(&self.pcm).unwrap_or_default();
        let all_frames = all.len() / self.n_mels;
        if all_frames > have {
            self.mel.extend_from_slice(
                &all[have * self.n_mels..all_frames * self.n_mels],
            );
        }
    }
}

/// Downmix interleaved multi-channel audio to mono 16 kHz (linear
/// resample).
pub fn to_mono_16k(
    raw_audio: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Vec<f32> {
    let mono: Vec<f32> = if channels <= 1 {
        raw_audio.to_vec()
    } else {
        raw_audio
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if sample_rate != 16000 {
        let ratio = 16000.0 / sample_rate as f32;
        let out_len = (mono.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(mono.len() - 1);
            let frac = src_idx - idx0 as f32;
            resampled.push(mono[idx0] * (1.0 - frac) + mono[idx1] * frac);
        }
        resampled
    } else {
        mono
    }
}
