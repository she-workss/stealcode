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
    /// Twiddle factors for the real-input FFT: the even/odd packed
    /// half-size FFT (`n_fft / 2`).
    twiddles_half: Vec<f32>,
    /// Twiddle factors for the recombination step (`n_fft`).
    twiddles: Vec<f32>,
    fft_re: Vec<f32>,
    fft_im: Vec<f32>,
    /// Recombined spectrum `[0..=n_fft/2]` (full half-complex band).
    out_re: Vec<f32>,
    out_im: Vec<f32>,
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
        let half = n / 2;
        let mut twiddles = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            twiddles.push(angle.cos());
            twiddles.push(angle.sin());
        }
        let mut twiddles_half = Vec::with_capacity(half);
        for k in 0..half {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / half as f32;
            twiddles_half.push(angle.cos());
            twiddles_half.push(angle.sin());
        }

        Ok(Self {
            cfg,
            window,
            fb,
            twiddles,
            twiddles_half,
            fft_re: vec![0.0; n],
            fft_im: vec![0.0; n],
            out_re: vec![0.0; half + 1],
            out_im: vec![0.0; half + 1],
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
    /// The signal is real, so it uses the even/odd packing trick: one
    /// complex FFT of length n_fft/2 plus a per-bin recombination
    /// (Sorensen), which halves the transform work.
    fn fft_frame(&mut self, frame: &[f32]) {
        let n = self.cfg.n_fft;
        let half = n / 2;
        // z[m] = x[2m] * w[2m] + i * x[2m+1] * w[2m+1]
        for m in 0..half {
            self.fft_re[m] = frame[2 * m] * self.window[2 * m];
            self.fft_im[m] = frame[2 * m + 1] * self.window[2 * m + 1];
        }
        Self::fft_inplace(
            half,
            &mut self.fft_re[..half],
            &mut self.fft_im[..half],
            &self.twiddles_half,
        );
        // Recombine into the half-complex spectrum X[0..=n/2]:
        //   Xe[m] = (Z[m] + conj(Z[N-m])) / 2
        //   Xo[m] = -i (Z[m] - conj(Z[N-m])) / 2
        //   X[m]  = Xe[m] + W_n^m * Xo[m]   (indices mod N = half)
        let tw = &self.twiddles;
        for m in 0..=half {
            let zm_re = self.fft_re[m % half];
            let zm_im = self.fft_im[m % half];
            let zn_re = self.fft_re[(half - m) % half];
            let zn_im = self.fft_im[(half - m) % half];
            let xe_re = (zm_re + zn_re) * 0.5;
            let xe_im = (zm_im - zn_im) * 0.5;
            let xo_re = (zm_im + zn_im) * 0.5;
            let xo_im = -(zm_re - zn_re) * 0.5;
            // W_n^m; m == half wraps to W^(n/2) = -1 (index would be
            // out of the twiddle table).
            let (wr, wi) = if m < half {
                (tw[2 * m], tw[2 * m + 1])
            } else {
                (-1.0, 0.0)
            };
            self.out_re[m] = xe_re + wr * xo_re - wi * xo_im;
            self.out_im[m] = xe_im + wr * xo_im + wi * xo_re;
        }
        self.fft_re[..=half].copy_from_slice(&self.out_re[..=half]);
        self.fft_im[..=half].copy_from_slice(&self.out_im[..=half]);
    }

    /// Iterative radix-2 Cooley-Tukey with bit-reversal permutation.
    fn fft_inplace(len: usize, re: &mut [f32], im: &mut [f32], tw: &[f32]) {
        let n = len;
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for i in (0..n).step_by(len) {
                for k in 0..half {
                    let t_re = re[i + k + half] * tw[2 * k * step]
                        - im[i + k + half] * tw[2 * k * step + 1];
                    let t_im = re[i + k + half] * tw[2 * k * step + 1]
                        + im[i + k + half] * tw[2 * k * step];
                    re[i + k + half] = re[i + k] - t_re;
                    im[i + k + half] = im[i + k] - t_im;
                    re[i + k] += t_re;
                    im[i + k] += t_im;
                }
            }
            len <<= 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The real-input FFT must match the naive DFT of the same windowed
    /// frame (parity with the reference mel frontend).
    #[test]
    fn fft_matches_naive_dft() {
        let cfg = PreprocessorConfig {
            sample_rate: 16000,
            n_fft: 16,
            win_length: 16,
            hop: 4,
            n_mels: 2,
            preemph: 0.97,
            dither: 0.0,
        };
        let fb = vec![1.0; (16 / 2 + 1) * 2];
        let mut fe = MelFrontend::new(cfg, &fb).unwrap();
        let pcm: Vec<f32> = (0..64)
            .map(|i| ((i * 37) % 101) as f32 / 17.0 - 2.0)
            .collect();
        let n = 16;
        let half = 8;
        let window = fe.window.clone();
        let mut frame = vec![0.0f32; n];
        for t in 0..4 {
            frame.copy_from_slice(&pcm[t * 4..t * 4 + n]);
            fe.fft_frame(&frame);
            for k in 0..=half {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for i in 0..n {
                    let x = frame[i] as f64 * window[i] as f64;
                    let a = -2.0 * std::f64::consts::PI * k as f64 * i as f64
                        / n as f64;
                    re += x * a.cos();
                    im += x * a.sin();
                }
                let d_re = fe.fft_re[k] as f64 - re;
                let d_im = fe.fft_im[k] as f64 - im;
                let tol = 1e-3 * (1.0 + re.abs() + im.abs());
                assert!(
                    d_re.abs() < tol && d_im.abs() < tol,
                    "frame {t} bin {k}: fft ({}, {}) vs dft ({re:.6}, {im:.6})",
                    fe.fft_re[k],
                    fe.fft_im[k]
                );
            }
        }
    }
}
