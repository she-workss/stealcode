//! Real-time speech-to-text from system audio (loopback) — not the
//! microphone. Captures everything the computer is playing (music,
//! videos, calls, games) and prints the transcript inline as chunks
//! are recognized.
//!
//! Backends:
//! - Windows: WASAPI loopback on the default render endpoint.
//! - Linux: `PulseAudio` monitor of the default sink (`@DEFAULT_SINK@.monitor`,
//!   works with `PipeWire`'s `PulseAudio` compatibility). Needs `libpulse`
//!   development files to build, e.g. `sudo apt install libpulse-dev`.
//! - macOS 13+: `ScreenCaptureKit` system audio. Grant the binary
//!   "Screen Recording" permission in System Settings → Privacy & Security.
//!
//! Usage:
//!   `cargo run -p audio_engine --example live_desktop -- --model path/to/model.gguf`

use std::{
    io::{self, Write},
    path::Path,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use audio_engine::{
    Nemotron,
    dsp::to_mono_16k,
    model::{AsrModel, LatencyMode, LiveAsr},
};

/// How much audio a streaming encoder step covers, trading latency
/// against throughput.
const LATENCY_MODE: LatencyMode = LatencyMode::HighQuality;

/// Print only the newly appended transcript text (the committed prefix
/// is never re-printed or rewritten).
fn print_new(model: &Nemotron, tr: &dyn LiveAsr<Nemotron>, printed: &mut usize) {
    let text = tr.text(model);
    if text.len() > *printed {
        print!("{}", &text[*printed..]);
        io::stdout().flush().ok();
        *printed = text.len();
    }
}

/// Decode interleaved f32-LE bytes into mono 16 kHz and append to the
/// shared buffer.
fn push_interleaved_f32(buf: &Mutex<Vec<f32>>, bytes: &[u8], channels: usize, rate: u32) {
    let interleaved: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let mono = to_mono_16k(&interleaved, channels, rate);
    buf.lock().unwrap().extend_from_slice(&mono);
}

#[cfg(windows)]
fn start_capture(buf: Arc<Mutex<Vec<f32>>>, err_tx: mpsc::Sender<anyhow::Error>) {
    use wasapi::{
        DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat, initialize_mta,
    };
    if let Err(e) = (|| -> Result<()> {
        initialize_mta().ok()?;
        let enumerator = DeviceEnumerator::new()?;
        let device = enumerator.get_default_device(&Direction::Render)?;
        println!("Capturing system audio from: {}", device.get_friendlyname()?);
        let mut client = device.get_iaudioclient()?;
        let format = WaveFormat::new(32, 32, &SampleType::Float, 44_100, 2, None);
        let (_, min_period) = client.get_device_period()?;
        // A client on a render endpoint initialized as a shared-mode
        // capture stream becomes a loopback stream (wasapi sets
        // AUDCLNT_STREAMFLAGS_LOOPBACK for us).
        client.initialize_client(
            &format,
            &Direction::Capture,
            &StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: min_period,
            },
        )?;
        let event = client.set_get_eventhandle()?;
        let capture = client.get_audiocaptureclient()?;
        client.start_stream()?;
        let blockalign = format.get_blockalign() as usize;
        let mut queue = std::collections::VecDeque::new();
        loop {
            // Sleep on a quiet stream instead of spinning on empty reads.
            if event.wait_for_event(1000).is_err() {
                thread::sleep(Duration::from_millis(10));
            }
            capture.read_from_device_to_deque(&mut queue)?;
            let frames = queue.len() / blockalign;
            if frames > 0 {
                let bytes: Vec<u8> = queue.drain(..frames * blockalign).collect();
                push_interleaved_f32(&buf, &bytes, 2, 44_100);
            }
        }
    })() {
        eprintln!("Loopback capture error: {e:#}");
        let _ = err_tx.send(e);
    }
}

#[cfg(target_os = "linux")]
fn start_capture(buf: Arc<Mutex<Vec<f32>>>, err_tx: mpsc::Sender<anyhow::Error>) {
    use libpulse_binding::sample::{Format, Spec};
    use libpulse_binding::stream::Direction;
    use libpulse_simple_binding::Simple;
    if let Err(e) = (|| -> Result<()> {
        let spec = Spec {
            format: Format::F32le,
            channels: 2,
            rate: 44_100,
        };
        let rec = Simple::new(
            None,
            "live_desktop",
            Direction::Record,
            Some("@DEFAULT_SINK@.monitor"), // monitor of the default sink
            "system audio",
            &spec,
            None,
            None,
        )?;
        println!("Capturing system audio via PulseAudio monitor (default sink)");
        let mut bytes = vec![0u8; 44_100 / 10 * 2 * 4]; // 100 ms
        loop {
            rec.read(&mut bytes)?;
            push_interleaved_f32(&buf, &bytes, 2, 44_100);
        }
    })() {
        eprintln!("Loopback capture error: {e:#}");
        let _ = err_tx.send(e);
    }
}

#[cfg(target_os = "macos")]
fn start_capture(buf: Arc<Mutex<Vec<f32>>>, err_tx: mpsc::Sender<anyhow::Error>) {
    use screencapturekit::cm::{AudioBuffer, AudioBufferList, CMSampleBufferExt};
    use screencapturekit::prelude::{
        SCContentFilter, SCShareableContent, SCStream, SCStreamConfiguration,
        SCStreamOutputTrait, SCStreamOutputType,
    };

    struct AudioHandler {
        buf: Arc<Mutex<Vec<f32>>>,
    }

    impl SCStreamOutputTrait for AudioHandler {
        fn did_output_sample_buffer(
            &self,
            sample: CMSampleBuffer,
            of_type: SCStreamOutputType,
        ) {
            if of_type != SCStreamOutputType::Audio {
                return;
            }
            let Some(list) = sample.audio_buffer_list() else {
                return;
            };
            // ScreenCaptureKit delivers one buffer per channel, f32-LE.
            let channels: Vec<&AudioBuffer> = list.iter().collect();
            let frames = channels
                .iter()
                .map(|ch| ch.data().len() / 4)
                .min()
                .unwrap_or(0);
            if frames == 0 {
                return;
            }
            let mut interleaved = Vec::with_capacity(frames * channels.len());
            for i in 0..frames {
                for ch in &channels {
                    let s = &ch.data()[i * 4..][..4];
                    interleaved.push(f32::from_le_bytes(s.try_into().unwrap()));
                }
            }
            let mono = to_mono_16k(&interleaved, channels.len(), 48_000);
            self.buf.lock().unwrap().extend_from_slice(&mono);
        }
    }

    if let Err(e) = (|| -> Result<()> {
        let content = SCShareableContent::get()?;
        let display = content
            .displays()
            .into_iter()
            .next()
            .context("no display found")?;
        let filter = SCContentFilter::create()
            .with_display(&display)
            .with_excluding_windows(&[])
            .build();
        let config = SCStreamConfiguration::new()
            .with_captures_audio(true)
            .with_sample_rate(48_000)
            .with_channel_count(2);
        let mut stream = SCStream::new(&filter, &config);
        stream.add_output_handler(AudioHandler { buf }, SCStreamOutputType::Audio);
        stream.start_capture()?;
        println!(
            "Capturing system audio via ScreenCaptureKit \
             (grant Screen Recording permission in System Settings)"
        );
        loop {
            thread::sleep(Duration::from_secs(3600));
        }
    })() {
        eprintln!("Loopback capture error: {e:#}");
        let _ = err_tx.send(e);
    }
}

fn main() -> Result<()> {
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    bail!("live_desktop captures system audio: WASAPI (Windows), PulseAudio monitor (Linux), ScreenCaptureKit (macOS)");

    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next(),
            other => bail!("unknown arg {other}"),
        }
    }
    let model_path = model_path.with_context(|| "usage: live_desktop --model <model.gguf>")?;
    println!("Loading model...");
    let mut model = Nemotron::load(Path::new(&model_path))?;
    println!("Model loaded\n");
    // Unbounded shared buffer: the capture thread never blocks and no
    // audio is ever dropped, even while a batch is being encoded.
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let (err_tx, err_rx) = mpsc::channel::<anyhow::Error>();
    let cap_buf = Arc::clone(&buf);
    thread::Builder::new()
        .name("loopback-capture".into())
        .spawn(move || start_capture(cap_buf, err_tx))?;
    println!("Live transcription. Play something on the computer.");
    let mut tr = model.live(LATENCY_MODE)?;
    let mut printed = 0usize;
    loop {
        if let Ok(e) = err_rx.try_recv() {
            return Err(e).context("loopback capture failed");
        }
        let samples: Vec<f32> = {
            let mut b = buf.lock().unwrap();
            std::mem::take(&mut *b)
        };
        if !samples.is_empty() {
            tr.push(&mut model, &samples)?;
            print_new(&model, tr.as_ref(), &mut printed);
        }
        thread::sleep(Duration::from_millis(25));
    }
}