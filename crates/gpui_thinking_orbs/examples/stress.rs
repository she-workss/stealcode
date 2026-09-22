//! Multi-orb stress / measurement harness.
//!
//! ```bash
//! cargo run --release --example stress -- --count 12 --state working --size avatar
//! cargo run --release --example stress -- --count 0     # control: empty window
//! ```
//!
//! Prints its own CPU usage (% of one core) and RSS every 2 s, so no external
//! profiler is needed.

use gpui::{
    App, Bounds, Context, Window, WindowBounds, WindowOptions, div, prelude::*,
    px, rgb, size,
};
use gpui_thinking_orbs::{OrbSize, OrbState, OrbTheme, ThinkingOrb};

struct Args {
    count: usize,
    state: OrbState,
    size: OrbSize,
    fps: f32,
    /// Keep animating even though the harness window never takes focus.
    /// Without this every measurement collapses to the idle baseline.
    pause_when_inactive: bool,
    /// Render a frozen grid of every state/size for pixel diffing.
    golden: bool,
}

fn parse_args() -> Args {
    let mut count = 1usize;
    let mut state = OrbState::Working;
    let mut size = OrbSize::Avatar;
    let mut fps = gpui_thinking_orbs::DEFAULT_TARGET_FPS;
    let mut pause_when_inactive = false;
    let mut golden = false;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--count" => {
                count =
                    argv.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1);
                i += 1;
            }
            "--fps" => {
                fps =
                    argv.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(fps);
                i += 1;
            }
            "--pause-when-inactive" => pause_when_inactive = true,
            "--golden" => golden = true,
            "--state" => {
                let name = argv.get(i + 1).cloned().unwrap_or_default();
                state = OrbState::ALL_STATES
                    .iter()
                    .copied()
                    .find(|s| s.as_str() == name)
                    .unwrap_or(OrbState::Working);
                i += 1;
            }
            "--size" => {
                size = OrbSize::ALL_SIZES
                    .iter()
                    .copied()
                    .find(|size| {
                        Some(size.as_str())
                            == argv.get(i + 1).map(String::as_str)
                    })
                    .unwrap_or(OrbSize::Avatar);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    Args {
        count,
        state,
        size,
        fps,
        pause_when_inactive,
        golden,
    }
}

/// Self-report CPU (% of one core) + RSS every 2 s by reading `/proc/self`.
fn spawn_reporter(label: String) {
    std::thread::spawn(move || {
        let hz = 100.0_f64; // CLK_TCK on Linux
        let read_cpu = || -> f64 {
            let s =
                std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
            // utime = field 14, stime = field 15 (1-indexed) - but comm may
            // contain spaces, so split after the closing ')'.
            let tail = s.rsplit_once(')').map_or(s.as_str(), |(_, t)| t);
            let f: Vec<&str> = tail.split_whitespace().collect();
            let utime: f64 =
                f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let stime: f64 =
                f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
            (utime + stime) / hz
        };
        let read_rss = || -> String {
            std::fs::read_to_string("/proc/self/status")
                .unwrap_or_default()
                .lines()
                .find(|l| l.starts_with("VmRSS"))
                .map(|l| l.trim_start_matches("VmRSS:").trim().to_string())
                .unwrap_or_default()
        };
        // Warm-up: let the window map and settle before the first sample.
        std::thread::sleep(std::time::Duration::from_secs(3));
        let mut prev = read_cpu();
        let mut prev_ticks = TICKS.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let now = read_cpu();
            let ticks = TICKS.load(std::sync::atomic::Ordering::Relaxed);
            println!(
                "[{label}] cpu={:5.1}% of one core   rss={}   orb0={:5.1} fps",
                (now - prev) / 2.0 * 100.0,
                read_rss(),
                (ticks - prev_ticks) as f64 / 2.0,
            );
            prev = now;
            prev_ticks = ticks;
        }
    });
}

struct Stress {
    orbs: Vec<gpui::Entity<ThinkingOrb>>,
    /// Kept alive so the tick observer stays subscribed.
    _tick_probe: Option<gpui::Subscription>,
}

/// Counts notifications from orb #0, so a measurement can prove the orb was
/// actually animating rather than silently frozen.
static TICKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

impl Stress {
    fn new(args: &Args, cx: &mut Context<'_, Self>) -> Self {
        // `--golden` renders one frozen orb per state, at the deterministic
        // `reduced_motion` timestamp, so two builds can be screenshot and
        // diffed pixel-for-pixel to prove visual fidelity was preserved.
        let orbs: Vec<_> = if args.golden {
            OrbState::ALL_STATES
                .iter()
                .copied()
                .flat_map(|state| {
                    [OrbSize::Avatar, OrbSize::Inline]
                        .map(move |sz| (state, sz))
                })
                .map(|(state, sz)| {
                    cx.new(move |_| {
                        ThinkingOrb::new()
                            .state(state)
                            .size(sz)
                            .theme(OrbTheme::Dark)
                            .reduced_motion(true)
                    })
                })
                .collect()
        } else {
            (0..args.count)
                .map(|_| {
                    cx.new(|_| {
                        ThinkingOrb::new()
                            .state(args.state)
                            .size(args.size)
                            .theme(OrbTheme::Dark)
                            .target_fps(args.fps)
                            .pause_when_inactive(args.pause_when_inactive)
                    })
                })
                .collect()
        };
        let tick_probe = orbs.first().map(|orb| {
            cx.observe(orb, |_, _, _| {
                TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            })
        });
        Self {
            orbs,
            _tick_probe: tick_probe,
        }
    }
}

impl Render for Stress {
    fn render(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        div()
            .size_full()
            .bg(rgb(0x0009_090b))
            .flex()
            .flex_wrap()
            .gap_4()
            .p_4()
            .children(self.orbs.iter().cloned())
    }
}

fn main() {
    let args = parse_args();
    spawn_reporter(format!(
        "count={} {} @{}fps",
        args.count,
        args.state.as_str(),
        args.fps
    ));
    gpui_kit::application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(640.), px(480.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|cx| Stress::new(&args, cx)),
        )
        .unwrap();
        cx.activate(true);
    });
}
