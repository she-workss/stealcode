use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender},
    },
    time::Duration,
};

use anyhow::Context as _;
#[cfg(feature = "voice")]
use gpui::Entity;
use gpui::{
    App, AsyncApp, Bounds, Context, IntoElement, Render, SharedString,
    TitlebarOptions, Window, WindowBounds, WindowHandle, WindowOptions, div,
    point, prelude::*, px, size,
};
#[cfg(feature = "voice")]
use gpui_component::button::ButtonVariants;
#[cfg(feature = "voice")]
use gpui_component::input::{Input, InputState};
use gpui_component::{Disableable, Root, StyledExt, button::Button};
use settings::Settings;
use sound::sounds::SoundName;
use tracing::error;
use tray_icon::{
    TrayIconBuilder, TrayIconEvent,
    menu::{Menu, MenuEvent, MenuItem},
};
#[cfg(feature = "voice")]
use voice::VoiceManager;

const APP_USER_MODEL_ID: &str = "he-thinks.StealCode";

/// How often the background task polls the voice/update workers for new
/// status text and asks the view to redraw (mirrors the TUI's poll loop).
const POLL_INTERVAL: Duration = Duration::from_millis(100);

#[cfg(target_os = "macos")]
static SOUND: &str = "Submarine";

#[cfg(all(unix, not(target_os = "macos")))]
static SOUND: &str = "message-new-instant";

#[cfg(target_os = "windows")]
static SOUND: &str = "Default";

#[cfg(target_os = "windows")]
fn show_notification() {
    let _ = notify_rust::Notification::new()
        .app_id(APP_USER_MODEL_ID)
        .summary("StealCode")
        .body("Hello from StealCode!")
        .sound_name(SOUND)
        .show();
}

#[cfg(not(target_os = "windows"))]
fn show_notification() {
    let _ = notify_rust::Notification::new()
        .summary("StealCode")
        .body("Hello from StealCode!")
        .sound_name(SOUND)
        .show();
}

/// Background worker for update checks / auto-update polling; owns the
/// channels and UI state (worker thread lives in
/// `auto_update::spawn_update_worker`, shared with the TUI).
#[derive(Debug)]
struct UpdateManager {
    auto_update_enabled: bool,
    status: String,
    /// Version an update is available for, from the last successful check.
    available_version: Option<semver::Version>,
    tx_cmd: Option<Sender<auto_update::UpdateWorkerCommand>>,
    rx_event: Option<Receiver<auto_update::UpdateWorkerEvent>>,
}

impl UpdateManager {
    fn new(auto_update_enabled: bool) -> Self {
        Self {
            auto_update_enabled,
            status: if auto_update_enabled {
                "Auto-update: ON".to_string()
            } else {
                "Auto-update: OFF (manual only)".to_string()
            },
            available_version: None,
            tx_cmd: None,
            rx_event: None,
        }
    }

    fn start_worker_if_needed(&mut self) {
        if self.tx_cmd.is_some() {
            return;
        }
        let current_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .unwrap_or_else(|_| semver::Version::new(0, 0, 0));
        let (tx_cmd, rx_event) = auto_update::spawn_update_worker(
            "she-workss",
            "stealcode",
            std::env::var("STEALCODE_GH_TOKEN").ok(),
            current_version,
            release_channel::ReleaseChannel::current(),
        );
        self.tx_cmd = Some(tx_cmd);
        self.rx_event = Some(rx_event);
    }

    fn check_now(&mut self) {
        self.start_worker_if_needed();
        self.status = "Checking...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(auto_update::UpdateWorkerCommand::Check);
        }
    }

    fn update_and_restart(&mut self) {
        if self.available_version.is_none() {
            return;
        }
        self.start_worker_if_needed();
        self.status = "Downloading update...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(auto_update::UpdateWorkerCommand::UpdateAndRestart);
        }
    }

    fn toggle_auto_update(&mut self) {
        self.auto_update_enabled = !self.auto_update_enabled;
        if self.status == "Checking..." {
            return;
        }
        self.status = if self.auto_update_enabled {
            "Auto-update: ON".to_string()
        } else {
            "Auto-update: OFF (manual only)".to_string()
        };
    }

    fn poll_events(&mut self) {
        use auto_update::UpdateWorkerEvent;
        if let Some(rx) = &self.rx_event {
            if let Ok(event) = rx.try_recv() {
                match event {
                    UpdateWorkerEvent::Status(status) => self.status = status,
                    UpdateWorkerEvent::Checked(version) => {
                        self.status = match &version {
                            Some(version) => {
                                format!("Update available: v{version}")
                            }
                            None => "Up to date".to_string(),
                        };
                        self.available_version = version;
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct StealcodeApp {
    #[cfg(feature = "voice")]
    voice: VoiceManager,
    /// Backing state for the read-only transcript textarea, kept in sync by
    /// the polling task via `set_value` (`Input` has no per-render override).
    #[cfg(feature = "voice")]
    voice_input: Entity<InputState>,
    updates: UpdateManager,
}

impl StealcodeApp {
    #[cfg_attr(not(feature = "voice"), allow(unused_variables))]
    fn new(window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        #[cfg(feature = "voice")]
        let voice_input = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .rows(4)
                .placeholder("Press \"Start voice recognition\" and speak...")
        });

        // VoiceManager works on a background thread and exposes state only
        // through `poll_events()` (as in the TUI), so poll it from a
        // lightweight task for the life of the view. `spawn_in`/`update_in`
        // are required because `InputState::set_value` needs a `&mut Window`.
        cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(POLL_INTERVAL).await;
                let alive = this.update_in(cx, |this, window, cx| {
                    this.updates.poll_events();
                    #[cfg(feature = "voice")]
                    {
                        this.voice.poll_events();
                        let text = &this.voice.text;
                        this.voice_input.update(cx, |input_state, cx| {
                            input_state.set_value(text, window, cx);
                        });
                    }
                    #[cfg(not(feature = "voice"))]
                    let _ = window;
                    cx.notify();
                });
                if alive.is_err() {
                    // The view and its window are gone; stop polling.
                    break;
                }
            }
        })
        .detach();

        let updates = UpdateManager::new(true);

        Self {
            #[cfg(feature = "voice")]
            voice: VoiceManager::new(),
            #[cfg(feature = "voice")]
            voice_input,
            updates,
        }
    }
}

impl Render for StealcodeApp {
    fn render(
        &mut self,
        _: &mut Window,
        cx: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        #[cfg(feature = "voice")]
        let is_recording = self.voice.is_recording;
        #[cfg(feature = "voice")]
        let toggle_label = if is_recording {
            "Stop voice recognition"
        } else {
            "Start voice recognition"
        };

        div()
            .v_flex()
            .gap_2()
            .size_full()
            .items_center()
            .justify_center()
            .child("Hello, World!")
            .child(
                Button::new("show_notification_btn")
                    .label("Show notification")
                    .on_click(|_, _, _| {
                        show_notification();
                    }),
            )
            .child(div().flex().flex_wrap().gap_2().justify_center().children(
                SoundName::ALL.into_iter().map(|sound| {
                    Button::new(sound.label())
                        .label(sound.label())
                        .on_click(move |_, _, _| sound::engine::play(sound))
                }),
            ))
            .child({
                #[cfg(feature = "voice")]
                {
                    div()
                        .v_flex()
                        .gap_2()
                        .w(px(420.))
                        .child(
                            Button::new("voice_toggle_btn")
                                .label(toggle_label)
                                .when(is_recording, |btn| btn.danger())
                                .on_click(cx.listener(|this, _, _window, cx| {
                                    this.voice.toggle();
                                    cx.notify();
                                })),
                        )
                        .child(SharedString::from(&self.voice.status))
                        .child(
                            // Kept in sync by the polling task's `set_value` above; `Input` has
                            // no per-render value override.
                            Input::new(&self.voice_input),
                        )
                }
                #[cfg(not(feature = "voice"))]
                div()
            })
            .child(
                div()
                    .v_flex()
                    .gap_2()
                    .w(px(420.))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .justify_center()
                            .child(
                                Button::new("check_update_btn")
                                    .label("Check for updates")
                                    .on_click(cx.listener(
                                        |this, _, _window, cx| {
                                            this.updates.check_now();
                                            cx.notify();
                                        },
                                    )),
                            )
                            .child(
                                Button::new("update_restart_btn")
                                    .label(
                                        if let Some(version) =
                                            &self.updates.available_version
                                        {
                                            format!(
                                                "Update and restart (v{version})"
                                            )
                                        } else {
                                            "Update and restart".to_string()
                                        },
                                    )
                                    .disabled(
                                        self.updates.available_version
                                            .is_none(),
                                    )
                                    .on_click(cx.listener(
                                        |this, _, _window, cx| {
                                            this.updates.update_and_restart();
                                            cx.notify();
                                        },
                                    )),
                            )
                            .child(
                                Button::new("toggle_auto_update_btn")
                                    .label(
                                        if self.updates.auto_update_enabled {
                                            "Auto-update: ON"
                                        } else {
                                            "Auto-update: OFF"
                                        },
                                    )
                                    .on_click(cx.listener(
                                        |this, _, _window, cx| {
                                            this.updates.toggle_auto_update();
                                            cx.notify();
                                        },
                                    )),
                            ),
                    )
                    .child(SharedString::from(&self.updates.status)),
            )
    }
}

#[derive(Debug)]
struct DummyView;

impl Render for DummyView {
    fn render(
        &mut self,
        _: &mut Window,
        _: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        div()
    }
}

fn open_app_window(
    window_handle: &Arc<Mutex<Option<WindowHandle<Root>>>>,
    cx: &mut App,
) {
    let mut state = window_handle.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(handle) = state.as_ref() {
        let is_alive = handle
            .update(cx, |_, window, _| {
                window.activate_window();
            })
            .is_ok();

        if is_alive {
            return;
        }
    }
    let result = cx.open_window(
        WindowOptions {
            titlebar: Some(TitlebarOptions {
                title: Some(SharedString::from("StealCode")),
                appears_transparent: false,
                traffic_light_position: None,
            }),
            ..Default::default()
        },
        |window, cx| {
            let view = cx.new(|cx| StealcodeApp::new(window, cx));
            cx.new(|cx| Root::new(view, window, cx))
        },
    );
    match result {
        Ok(handle) => {
            *state = Some(handle);
        }
        Err(e) => error!("failed to open window: {e:?}"),
    }
}

fn load_icon() -> anyhow::Result<tray_icon::Icon> {
    // Compiled in so the binary doesn't depend on the build machine's path;
    // `env!("CARGO_MANIFEST_DIR")` would bake that path in.
    let image_bytes = include_bytes!("../../cli/assets/icons/prod/icon.png");
    let image = image::load_from_memory(image_bytes)
        .context("failed to decode embedded icon")?
        .into_rgba8();
    let (width, height) = image.dimensions();
    tray_icon::Icon::from_rgba(image.into_raw(), width, height)
        .context("failed to open icon")
}

#[cfg(target_os = "windows")]
fn setup_windows_app_id() {
    use windows::{
        Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID, core::w,
    };
    #[allow(unsafe_code)]
    let _ = unsafe {
        SetCurrentProcessExplicitAppUserModelID(w!("he-thinks.StealCode"))
    };
}

pub fn run_desktop(
    _settings: &Settings,
    _project: Option<&Path>,
) -> anyhow::Result<()> {
    #[cfg(target_os = "windows")]
    setup_windows_app_id();
    let icon = load_icon()?;
    // Apply a staged update (e.g. from `stealcode upgrade` or another
    // instance) via the helper, which relaunches StealCode; exit this process.
    #[cfg(target_os = "windows")]
    if auto_update::apply_staged_update_on_startup()? {
        std::process::exit(0);
    }
    gpui_platform::application().run(move |cx| {
        // Apply a silent update finished while running as we exit. The
        // subscription must stay alive for the hook (same leak trick as the
        // tray icon); renaming a running exe is legal on Windows, so the
        // helper's retry loop rides out our shutdown.
        #[cfg(target_os = "windows")]
        std::mem::forget(cx.on_app_quit(|_| async move {
            auto_update::finalize_auto_update_on_quit().await;
        }));
        // Remove stale update dirs from a crashed previous update at
        // startup (only empty dirs; see `cleanup_windows`).
        #[cfg(target_os = "windows")]
        cx.spawn(|_: &mut AsyncApp| async move {
            if let Err(error) = auto_update::cleanup_windows().await {
                error!("failed to clean up update dirs: {error}");
            }
        })
        .detach();
        gpui_component::init(cx);
        let result = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds {
                    origin: point(px(0.), px(0.)),
                    size: size(px(1.), px(1.)),
                })),
                titlebar: None,
                show: false,
                ..Default::default()
            },
            |_window, cx| cx.new(|_| DummyView),
        );
        if let Err(error) = result {
            error!("failed to create dummy window: {error:?}");
        }
        let main_window_handle: Arc<Mutex<Option<WindowHandle<Root>>>> =
            Arc::new(Mutex::new(None));
        let menu = Menu::new();
        let show_notif_item =
            MenuItem::with_id("show_notif", "Show notification", true, None);
        let exit_item = MenuItem::with_id("exit", "Exit", true, None);
        let _ = menu.append(&show_notif_item);
        let _ = menu.append(&exit_item);
        let tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("StealCode")
            .with_icon(icon)
            .with_menu_on_left_click(false)
            .build();
        match tray_icon {
            Ok(tray_icon) => std::mem::forget(tray_icon),
            Err(error) => error!("failed to create tray icon: {error}"),
        }
        cx.spawn({
            let main_window_handle = main_window_handle.clone();
            async move |cx: &mut AsyncApp| {
                loop {
                    if let Ok(event) = TrayIconEvent::receiver().try_recv() {
                        if let tray_icon::TrayIconEvent::Click {
                            button: tray_icon::MouseButton::Left,
                            button_state: tray_icon::MouseButtonState::Up,
                            ..
                        } = event
                        {
                            let handle = main_window_handle.clone();
                            let _ = cx.update(move |cx| {
                                open_app_window(&handle, cx);
                            });
                        }
                    }
                    if let Ok(event) = MenuEvent::receiver().try_recv() {
                        match event.id().as_ref() {
                            "show_notif" => {
                                show_notification();
                            }
                            "exit" => {
                                let _ = cx.update(|cx| {
                                    cx.quit();
                                });
                            }
                            _ => {}
                        }
                    }
                    cx.background_executor()
                        .timer(Duration::from_millis(50))
                        .await;
                }
            }
        })
        .detach();
        open_app_window(&main_window_handle, cx);
    });
    Ok(())
}
