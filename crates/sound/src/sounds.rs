#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

#[derive(Debug, Clone, Copy)]
pub struct ToneLayer {
    pub offset: f32,
    pub attack: f32,
    pub decay: f32,
    pub peak: f32,
    pub waveform: Waveform,
    pub frequency: f32,
    pub detune_cents: f32,
    pub glide_to: Option<f32>,
    pub glide_time: Option<f32>,
}

impl Default for ToneLayer {
    fn default() -> Self {
        Self {
            offset: 0.0,
            attack: 0.0,
            decay: 0.0,
            peak: 0.0,
            waveform: Waveform::Sine,
            frequency: 440.0,
            detune_cents: 0.0,
            glide_to: None,
            glide_time: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    LowPass,
    BandPass,
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseLayer {
    pub offset: f32,
    pub attack: f32,
    pub decay: f32,
    pub peak: f32,
    pub filter_type: FilterType,
    pub filter_frequency: f32,
    pub filter_q: f32,
}

impl Default for NoiseLayer {
    fn default() -> Self {
        Self {
            offset: 0.0,
            attack: 0.0,
            decay: 0.0,
            peak: 0.0,
            filter_type: FilterType::LowPass,
            filter_frequency: 1000.0,
            filter_q: 0.707,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SoundLayer {
    Tone(ToneLayer),
    Noise(NoiseLayer),
}

impl SoundLayer {
    #[must_use]
    pub const fn offset(&self) -> f32 {
        match self {
            Self::Tone(t) => t.offset,
            Self::Noise(n) => n.offset,
        }
    }

    #[must_use]
    pub const fn attack(&self) -> f32 {
        match self {
            Self::Tone(t) => t.attack,
            Self::Noise(n) => n.attack,
        }
    }

    #[must_use]
    pub const fn decay(&self) -> f32 {
        match self {
            Self::Tone(t) => t.decay,
            Self::Noise(n) => n.decay,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shimmer {
    pub delay: f32,
    pub feedback: f32,
    pub wet: f32,
    pub lowpass: f32,
}

#[derive(Debug, Clone)]
pub struct SoundRecipe {
    pub master_gain: f32,
    pub layers: Vec<SoundLayer>,
    pub shimmer: Option<Shimmer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoundName {
    Chime,
    Sparkle,
    Droplet,
    Bloom,
    Whisper,
    Tick,
    Press,
    Release,
    Toggle,
    Success,
    Error,
    Page,
    Loading,
    Ready,
    Pulse,
    Scan,
    Arrival,
}

impl SoundName {
    pub const ALL: [Self; 17] = [
        Self::Chime,
        Self::Sparkle,
        Self::Droplet,
        Self::Bloom,
        Self::Whisper,
        Self::Tick,
        Self::Press,
        Self::Release,
        Self::Toggle,
        Self::Success,
        Self::Error,
        Self::Page,
        Self::Loading,
        Self::Ready,
        Self::Pulse,
        Self::Scan,
        Self::Arrival,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Chime => "chime",
            Self::Sparkle => "sparkle",
            Self::Droplet => "droplet",
            Self::Bloom => "bloom",
            Self::Whisper => "whisper",
            Self::Tick => "tick",
            Self::Press => "press",
            Self::Release => "release",
            Self::Toggle => "toggle",
            Self::Success => "success",
            Self::Error => "error",
            Self::Page => "page",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Pulse => "pulse",
            Self::Scan => "scan",
            Self::Arrival => "arrival",
        }
    }

    #[must_use]
    pub fn recipe(self) -> SoundRecipe {
        use SoundLayer::{Noise, Tone};
        match self {
            Self::Chime => SoundRecipe {
                master_gain: 0.5,
                layers: vec![
                    Tone(ToneLayer {
                        frequency: 1046.5,
                        attack: 0.006,
                        decay: 0.22,
                        peak: 0.09,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 1568.0,
                        offset: 0.09,
                        attack: 0.006,
                        decay: 0.26,
                        peak: 0.08,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.12,
                    feedback: 0.25,
                    wet: 0.18,
                    lowpass: 4000.0,
                }),
            },
            Self::Sparkle => SoundRecipe {
                master_gain: 0.5,
                layers: vec![
                    Tone(ToneLayer {
                        frequency: 1760.0,
                        attack: 0.003,
                        decay: 0.09,
                        peak: 0.045,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 2217.0,
                        offset: 0.045,
                        attack: 0.003,
                        decay: 0.09,
                        peak: 0.04,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 2637.0,
                        offset: 0.09,
                        attack: 0.003,
                        decay: 0.1,
                        peak: 0.038,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 3520.0,
                        offset: 0.135,
                        attack: 0.003,
                        decay: 0.12,
                        peak: 0.032,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.07,
                    feedback: 0.35,
                    wet: 0.22,
                    lowpass: 6000.0,
                }),
            },
            Self::Droplet => SoundRecipe {
                master_gain: 0.55,
                layers: vec![Tone(ToneLayer {
                    frequency: 1200.0,
                    glide_to: Some(550.0),
                    glide_time: Some(0.14),
                    attack: 0.004,
                    decay: 0.2,
                    peak: 0.075,
                    ..Default::default()
                })],
                shimmer: Some(Shimmer {
                    delay: 0.09,
                    feedback: 0.2,
                    wet: 0.15,
                    lowpass: 3000.0,
                }),
            },
            Self::Bloom => SoundRecipe {
                master_gain: 0.5,
                layers: vec![
                    Tone(ToneLayer {
                        frequency: 528.0,
                        attack: 0.06,
                        decay: 0.32,
                        peak: 0.06,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 528.0,
                        detune_cents: 12.0,
                        attack: 0.06,
                        decay: 0.34,
                        peak: 0.05,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.15,
                    feedback: 0.2,
                    wet: 0.12,
                    lowpass: 2500.0,
                }),
            },
            Self::Whisper => SoundRecipe {
                master_gain: 0.48,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::LowPass,
                        filter_frequency: 1600.0,
                        filter_q: 0.7,
                        attack: 0.025,
                        decay: 0.13,
                        peak: 0.04,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 880.0,
                        glide_to: Some(660.0),
                        glide_time: Some(0.14),
                        offset: 0.01,
                        attack: 0.012,
                        decay: 0.14,
                        peak: 0.025,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Tick => SoundRecipe {
                master_gain: 0.4,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 5400.0,
                        filter_q: 1.8,
                        attack: 0.001,
                        decay: 0.018,
                        peak: 0.14,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 2600.0,
                        attack: 0.001,
                        decay: 0.012,
                        peak: 0.018,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Press => SoundRecipe {
                master_gain: 0.4,
                layers: vec![Noise(NoiseLayer {
                    filter_type: FilterType::BandPass,
                    filter_frequency: 1700.0,
                    filter_q: 1.4,
                    attack: 0.001,
                    decay: 0.02,
                    peak: 0.13,
                    ..Default::default()
                })],
                shimmer: None,
            },
            Self::Release => SoundRecipe {
                master_gain: 0.4,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 4600.0,
                        filter_q: 1.8,
                        attack: 0.001,
                        decay: 0.016,
                        peak: 0.12,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 3200.0,
                        offset: 0.006,
                        attack: 0.001,
                        decay: 0.05,
                        peak: 0.02,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Toggle => SoundRecipe {
                master_gain: 0.4,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 2200.0,
                        filter_q: 1.6,
                        attack: 0.001,
                        decay: 0.016,
                        peak: 0.12,
                        ..Default::default()
                    }),
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 3800.0,
                        filter_q: 1.6,
                        offset: 0.024,
                        attack: 0.001,
                        decay: 0.02,
                        peak: 0.1,
                    }),
                ],
                shimmer: None,
            },
            Self::Success => SoundRecipe {
                master_gain: 0.5,
                layers: vec![
                    Tone(ToneLayer {
                        frequency: 880.0,
                        attack: 0.004,
                        decay: 0.09,
                        peak: 0.06,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 1108.73,
                        offset: 0.06,
                        attack: 0.004,
                        decay: 0.1,
                        peak: 0.06,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 1318.51,
                        offset: 0.12,
                        attack: 0.004,
                        decay: 0.18,
                        peak: 0.07,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.1,
                    feedback: 0.22,
                    wet: 0.16,
                    lowpass: 4500.0,
                }),
            },
            Self::Error => SoundRecipe {
                master_gain: 0.42,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 850.0,
                        filter_q: 1.1,
                        attack: 0.001,
                        decay: 0.035,
                        peak: 0.13,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        waveform: Waveform::Triangle,
                        frequency: 440.0,
                        offset: 0.025,
                        attack: 0.004,
                        decay: 0.09,
                        peak: 0.045,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        waveform: Waveform::Triangle,
                        frequency: 349.23,
                        offset: 0.1,
                        attack: 0.004,
                        decay: 0.14,
                        peak: 0.04,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Page => SoundRecipe {
                master_gain: 0.38,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::LowPass,
                        filter_frequency: 1800.0,
                        filter_q: 0.7,
                        attack: 0.006,
                        decay: 0.08,
                        peak: 0.11,
                        ..Default::default()
                    }),
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 4200.0,
                        filter_q: 1.2,
                        offset: 0.04,
                        attack: 0.004,
                        decay: 0.065,
                        peak: 0.08,
                    }),
                    Tone(ToneLayer {
                        frequency: 2400.0,
                        offset: 0.075,
                        attack: 0.002,
                        decay: 0.045,
                        peak: 0.02,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Loading => SoundRecipe {
                master_gain: 0.42,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::LowPass,
                        filter_frequency: 1400.0,
                        filter_q: 0.6,
                        attack: 0.035,
                        decay: 0.14,
                        peak: 0.035,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 420.0,
                        glide_to: Some(630.0),
                        glide_time: Some(0.18),
                        attack: 0.025,
                        decay: 0.18,
                        peak: 0.05,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.11,
                    feedback: 0.18,
                    wet: 0.12,
                    lowpass: 2800.0,
                }),
            },
            Self::Ready => SoundRecipe {
                master_gain: 0.48,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 3600.0,
                        filter_q: 1.8,
                        attack: 0.001,
                        decay: 0.02,
                        peak: 0.11,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        waveform: Waveform::Triangle,
                        frequency: 330.0,
                        glide_to: Some(660.0),
                        glide_time: Some(0.12),
                        offset: 0.012,
                        attack: 0.004,
                        decay: 0.16,
                        peak: 0.055,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 990.0,
                        offset: 0.13,
                        attack: 0.004,
                        decay: 0.22,
                        peak: 0.06,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.1,
                    feedback: 0.16,
                    wet: 0.1,
                    lowpass: 4200.0,
                }),
            },
            Self::Pulse => SoundRecipe {
                master_gain: 0.42,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::BandPass,
                        filter_frequency: 2600.0,
                        filter_q: 2.4,
                        attack: 0.001,
                        decay: 0.022,
                        peak: 0.08,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        waveform: Waveform::Triangle,
                        frequency: 620.0,
                        glide_to: Some(1240.0),
                        glide_time: Some(0.07),
                        attack: 0.002,
                        decay: 0.085,
                        peak: 0.055,
                        ..Default::default()
                    }),
                ],
                shimmer: None,
            },
            Self::Scan => SoundRecipe {
                master_gain: 0.4,
                layers: vec![
                    Tone(ToneLayer {
                        frequency: 740.0,
                        attack: 0.002,
                        decay: 0.055,
                        peak: 0.05,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 1110.0,
                        offset: 0.045,
                        attack: 0.002,
                        decay: 0.055,
                        peak: 0.045,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 1665.0,
                        offset: 0.09,
                        attack: 0.002,
                        decay: 0.07,
                        peak: 0.04,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.065,
                    feedback: 0.16,
                    wet: 0.1,
                    lowpass: 4200.0,
                }),
            },
            Self::Arrival => SoundRecipe {
                master_gain: 0.44,
                layers: vec![
                    Noise(NoiseLayer {
                        filter_type: FilterType::LowPass,
                        filter_frequency: 900.0,
                        filter_q: 0.8,
                        attack: 0.05,
                        decay: 0.24,
                        peak: 0.035,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 220.0,
                        glide_to: Some(440.0),
                        glide_time: Some(0.32),
                        attack: 0.04,
                        decay: 0.34,
                        peak: 0.055,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 659.25,
                        offset: 0.12,
                        attack: 0.045,
                        decay: 0.32,
                        peak: 0.04,
                        ..Default::default()
                    }),
                    Tone(ToneLayer {
                        frequency: 987.77,
                        offset: 0.19,
                        attack: 0.045,
                        decay: 0.34,
                        peak: 0.032,
                        ..Default::default()
                    }),
                ],
                shimmer: Some(Shimmer {
                    delay: 0.16,
                    feedback: 0.28,
                    wet: 0.18,
                    lowpass: 3200.0,
                }),
            },
        }
    }
}
