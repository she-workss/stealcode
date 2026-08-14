use std::{
    collections::HashMap,
    num::NonZero,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use rodio::{DeviceSinkBuilder, MixerDeviceSink, buffer::SamplesBuffer};

use crate::sounds::{
    FilterType, NoiseLayer, Shimmer, SoundLayer, SoundName, SoundRecipe,
    ToneLayer, Waveform,
};

const SAMPLE_RATE: u32 = 44_100;
const SOURCE_STOP_PADDING: f32 = 0.05;
const ENVELOPE_FLOOR: f32 = 0.0001;
const INAUDIBLE_GAIN: f32 = 0.001;
const CHANNELS: NonZero<u16> = NonZero::new(1u16).expect("1 is non-zero");
const SAMPLE_RATE_NONZERO: NonZero<u32> =
    NonZero::new(SAMPLE_RATE).expect("44100 is non-zero");

static ENABLED: AtomicBool = AtomicBool::new(true);
static RENDER_CACHE: LazyLock<Mutex<HashMap<SoundName, Arc<Vec<f32>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);
static STREAM: Mutex<Option<MixerDeviceSink>> = Mutex::new(None);

pub fn set_enabled(value: bool) {
    ENABLED.store(value, Ordering::Relaxed);
}

struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new(
        filter_type: FilterType,
        freq: f32,
        sample_rate: f32,
        q: f32,
    ) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q.max(0.0001));
        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            FilterType::LowPass => (
                (1.0 - cos_w0) / 2.0,
                1.0 - cos_w0,
                (1.0 - cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            FilterType::BandPass => {
                (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
            }
        };
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

struct Xorshift32(u32);

impl Xorshift32 {
    fn seeded() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0x9e3779b9)
            .max(1);
        Self(seed)
    }

    fn next_signed(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn exp_ramp(from: f32, to: f32, t: f32, duration: f32) -> f32 {
    if duration <= 0.0 {
        return to;
    }
    let t = t.clamp(0.0, duration);
    from * (to / from).powf(t / duration)
}

fn envelope(attack: f32, decay: f32, peak: f32, t: f32) -> f32 {
    if t < 0.0 {
        0.0
    } else if t <= attack {
        exp_ramp(ENVELOPE_FLOOR, peak.max(ENVELOPE_FLOOR), t, attack)
    } else if t <= attack + decay {
        exp_ramp(peak.max(ENVELOPE_FLOOR), ENVELOPE_FLOOR, t - attack, decay)
    } else {
        0.0
    }
}

fn waveform_sample(waveform: Waveform, phase: f32) -> f32 {
    let phase = phase.rem_euclid(std::f32::consts::TAU);
    match waveform {
        Waveform::Sine => phase.sin(),
        Waveform::Square => {
            if phase < std::f32::consts::PI {
                1.0
            } else {
                -1.0
            }
        }
        Waveform::Sawtooth => phase / std::f32::consts::PI - 1.0,
        Waveform::Triangle => {
            let x = phase / std::f32::consts::TAU;
            4.0 * (x - (x + 0.5).floor()).abs() - 1.0
        }
    }
}

fn source_end(recipe: &SoundRecipe) -> f32 {
    recipe
        .layers
        .iter()
        .map(|l| l.offset() + l.attack() + l.decay() + SOURCE_STOP_PADDING)
        .fold(0.0_f32, f32::max)
}

fn shimmer_tail(shimmer: Option<&Shimmer>) -> f32 {
    match shimmer {
        None => 0.0,
        Some(s) if s.feedback <= 0.0 => 0.0,
        Some(s) if s.feedback >= 1.0 => s.delay,
        Some(s) => {
            s.delay * (1.0 + (INAUDIBLE_GAIN.ln() / s.feedback.ln()).ceil())
        }
    }
}

fn render_tone(dry: &mut [f32], layer: &ToneLayer, sample_rate: f32) {
    let start = (layer.offset * sample_rate).round() as usize;
    let duration = layer.attack + layer.decay + SOURCE_STOP_PADDING;
    let len = (duration * sample_rate).ceil() as usize;
    let glide_time = layer.glide_time.unwrap_or(layer.attack + layer.decay);
    let detune_ratio = 2f32.powf(layer.detune_cents / 1200.0);
    let mut phase = 0.0_f32;
    for i in 0..len {
        let idx = start + i;
        if idx >= dry.len() {
            break;
        }
        let t = i as f32 / sample_rate;
        let base_freq = match layer.glide_to {
            Some(target) => exp_ramp(layer.frequency, target, t, glide_time),
            None => layer.frequency,
        };
        phase += std::f32::consts::TAU * base_freq * detune_ratio / sample_rate;
        let env = envelope(layer.attack, layer.decay, layer.peak, t);
        dry[idx] += waveform_sample(layer.waveform, phase) * env;
    }
}

fn render_noise(dry: &mut [f32], layer: &NoiseLayer, sample_rate: f32) {
    let start = (layer.offset * sample_rate).round() as usize;
    let duration = layer.attack + layer.decay + SOURCE_STOP_PADDING;
    let len = (duration * sample_rate).ceil() as usize;
    let mut filter = Biquad::new(
        layer.filter_type,
        layer.filter_frequency,
        sample_rate,
        layer.filter_q,
    );
    let mut rng = Xorshift32::seeded();
    for i in 0..len {
        let idx = start + i;
        if idx >= dry.len() {
            break;
        }
        let t = i as f32 / sample_rate;
        let filtered = filter.process(rng.next_signed());
        let env = envelope(layer.attack, layer.decay, layer.peak, t);
        dry[idx] += filtered * env;
    }
}

fn render_recipe(recipe: &SoundRecipe) -> Vec<f32> {
    let sample_rate = SAMPLE_RATE as f32;
    let total_seconds = source_end(recipe)
        + shimmer_tail(recipe.shimmer.as_ref())
        + SOURCE_STOP_PADDING;
    let total_len = (total_seconds * sample_rate).ceil() as usize + 1;
    let mut dry = vec![0.0_f32; total_len];
    for layer in &recipe.layers {
        match layer {
            SoundLayer::Tone(t) => render_tone(&mut dry, t, sample_rate),
            SoundLayer::Noise(n) => render_noise(&mut dry, n, sample_rate),
        }
    }
    for sample in &mut dry {
        *sample *= recipe.master_gain;
    }
    let Some(shimmer) = &recipe.shimmer else {
        return dry;
    };
    let delay_samples = (shimmer.delay * sample_rate).round() as usize;
    let mut delay_line = dry.clone();
    let mut filter =
        Biquad::new(FilterType::LowPass, shimmer.lowpass, sample_rate, 1.0);
    let mut out = dry;
    for n in 0..total_len {
        let delayed = if n >= delay_samples {
            delay_line[n - delay_samples]
        } else {
            0.0
        };
        let filtered = filter.process(delayed);
        out[n] += filtered * shimmer.wet;
        let future = n + delay_samples;
        if future < total_len {
            delay_line[future] += filtered * shimmer.feedback;
        }
    }
    out
}

fn rendered(sound: SoundName) -> Arc<Vec<f32>> {
    let mut cache = RENDER_CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache
        .entry(sound)
        .or_insert_with(|| Arc::new(render_recipe(&sound.recipe())))
        .clone()
}

fn open_stream() -> Option<MixerDeviceSink> {
    let builder = DeviceSinkBuilder::from_default_device()
        .ok()?
        .with_error_callback(|_err| {
            DEVICE_LOST.store(true, Ordering::Relaxed);
        });
    let mut sink = builder.open_sink_or_fallback().ok()?;
    sink.log_on_drop(false);
    Some(sink)
}

pub fn play(sound: SoundName) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut guard = STREAM
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_none() || DEVICE_LOST.swap(false, Ordering::Relaxed) {
        *guard = open_stream();
    }
    let Some(stream) = guard.as_ref() else {
        return;
    };
    let samples = Arc::unwrap_or_clone(rendered(sound));
    let buffer = SamplesBuffer::new(CHANNELS, SAMPLE_RATE_NONZERO, samples);
    stream.mixer().add(buffer);
}
