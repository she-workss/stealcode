use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender},
    },
    time::Duration,
};

use gpui::{
    App, AsyncApp, Bounds, Context, Entity, IntoElement, Render, SharedString,
    TitlebarOptions, Window, WindowBounds, WindowHandle, WindowOptions, div,
    point, prelude::*, px, size,
};
use gpui_component::{
    Disableable, Root, StyledExt,
    button::{Button, ButtonVariants},
    input::{Input, InputState},
};
use settings::Settings;
use sound::sounds::SoundName;
use tracing::error;
use tray_icon::{
    TrayIconBuilder, TrayIconEvent,
    menu::{Menu, MenuEvent, MenuItem},
};
use voice::VoiceManager;

const APP_USER_MODEL_ID: &str = "he-thinks.StealCode";

/// How often the background task checks `VoiceManager` for new status/text
/// updates and asks the view to redraw. Mirrors the TUI's per-loop-iteration
/// `state.voice.poll_events()` call, just adapted to GPUI's async model
/// instead of a blocking terminal event loop.
const VOICE_POLL_INTERVAL: Duration = Duration::from_millis(100);

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

/// Background worker for "Check for updates now" / auto-update polling,
/// used only to let the GUI exercise both the manual and automatic paths
/// through `auto_update` for testing. Same background-thread-plus-channel
/// shape as `VoiceManager`, kept separate from it since the two have
/// nothing in common besides that shape.
#[derive(Debug)]
enum UpdateCommand {
    Check,
    UpdateAndRestart,
}

#[derive(Debug)]
enum UpdateEvent {
    Status(String),
    Checked(Option<semver::Version>),
}

#[derive(Debug)]
struct UpdateManager {
    auto_update_enabled: bool,
    status: String,
    /// Version an update is available for, from the last successful check.
    available_version: Option<semver::Version>,
    tx_cmd: Option<Sender<UpdateCommand>>,
    rx_event: Option<Receiver<UpdateEvent>>,
}

impl UpdateManager {
    fn new(auto_update_enabled: bool) -> Self {
        Self {
            auto_update_enabled,
            status: Self::status_label(auto_update_enabled),
            available_version: None,
            tx_cmd: None,
            rx_event: None,
        }
    }

    const fn status_label(auto_update_enabled: bool) -> String {
        String::new() // placeholder, overwritten immediately below
    }

    fn start_worker_if_needed(&mut self) {
        if self.tx_cmd.is_some() {
            return;
        }
        let (tx_cmd, rx_cmd) = std::sync::mpsc::channel::<UpdateCommand>();
        let (tx_event, rx_event) = std::sync::mpsc::channel::<UpdateEvent>();
        std::thread::spawn(move || {
            while let Ok(command) = rx_cmd.recv() {
                let current_version =
                    semver::Version::parse(env!("CARGO_PKG_VERSION"))
                        .unwrap_or_else(|_| semver::Version::new(0, 0, 0));
                let owner = "she-workss";
                let repo = "stealcode";
                let token = std::env::var("STEALCODE_GH_TOKEN").ok();
                let channel = release_channel::ReleaseChannel::current();
                match command {
                    UpdateCommand::Check => {
                        let result = auto_update::check_now_blocking(
                            owner,
                            repo,
                            token,
                            current_version,
                            channel,
                        );
                        let event = match result {
                            Ok(Some(version)) => {
                                UpdateEvent::Checked(Some(version))
                            }
                            Ok(None) => UpdateEvent::Checked(None),
                            Err(error) => UpdateEvent::Status(format!(
                                "Check failed: {error}"
                            )),
                        };
                        let _ = tx_event.send(event);
                    }
                    UpdateCommand::UpdateAndRestart => {
                        let _ = tx_event.send(UpdateEvent::Status(
                            "Downloading update...".to_string(),
                        ));
                        match auto_update::update_now_blocking(
                            owner,
                            repo,
                            token,
                            current_version,
                            channel,
                        ) {
                            Ok(version) => {
                                let _ = tx_event.send(UpdateEvent::Status(
                                    format!(
                                        "Update installed: v{version} \
                                         - restarting"
                                    ),
                                ));
                                if let Err(error) =
                                    auto_update::restart_updated_app()
                                {
                                    let _ = tx_event.send(UpdateEvent::Status(
                                        format!("Restart failed: {error}"),
                                    ));
                                }
                            }
                            Err(error) => {
                                let _ = tx_event.send(UpdateEvent::Status(
                                    format!("Update failed: {error}"),
                                ));
                            }
                        }
                    }
                }
            }
        });
        self.tx_cmd = Some(tx_cmd);
        self.rx_event = Some(rx_event);
    }

    fn check_now(&mut self) {
        self.start_worker_if_needed();
        self.status = "Checking...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(UpdateCommand::Check);
        }
    }

    fn update_and_restart(&mut self) {
        if self.available_version.is_none() {
            return;
        }
        self.start_worker_if_needed();
        self.status = "Downloading update...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(UpdateCommand::UpdateAndRestart);
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
        if let Some(rx) = &self.rx_event {
            if let Ok(event) = rx.try_recv() {
                match event {
                    UpdateEvent::Status(status) => self.status = status,
                    UpdateEvent::Checked(version) => {
                        self.available_version = version.clone();
                        self.status = match version {
                            Some(version) => {
                                format!("Update available: v{version}")
                            }
                            None => "Up to date".to_string(),
                        };
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct StealcodeApp {
    voice: VoiceManager,
    /// Backing state for the read-only transcript textarea. Its buffer is
    /// written to from the polling task below via `set_value` whenever
    /// `VoiceManager::text` changes - there's no per-render override for
    /// `Input` the way some other gpui-component widgets have (e.g.
    /// `Clipboard::value`), so this is the only way to keep it in sync.
    voice_input: Entity<InputState>,
    updates: UpdateManager,
}

impl StealcodeApp {
    fn new(window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        let voice_input = cx.new(|cx| {
            InputState::new(window, cx)
                .multi_line(true)
                .rows(4)
                .placeholder("Press \"Start voice recognition\" and speak...")
        });

        // Same reasoning as the TUI's poll loop: VoiceManager runs its
        // actual work (mic capture, model load, transcription) on a
        // background thread and only exposes state through
        // `poll_events()`. GPUI has no equivalent of the TUI's blocking
        // event loop to hang this off of, so it gets its own lightweight
        // polling task instead, spawned once here and running for the
        // life of the view.
        //
        // `spawn_in`/`update_in` (rather than plain `spawn`/`update`) are
        // needed specifically because `InputState::set_value` requires a
        // `&mut Window`, not just a `Context` - plain `cx.spawn` only
        // hands back an `AsyncApp`, which has no window attached.
        cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(VOICE_POLL_INTERVAL).await;
                let alive = this.update_in(cx, |this, window, cx| {
                    this.voice.poll_events();
                    let text = this.voice.text.clone();
                    this.voice_input.update(cx, |input_state, cx| {
                        input_state.set_value(text, window, cx);
                    });
                    this.updates.poll_events();
                    cx.notify();
                });
                if alive.is_err() {
                    // The view (and its window) is gone - nothing left to
                    // update, stop polling instead of spinning forever.
                    break;
                }
            }
        })
        .detach();

        let mut updates = UpdateManager::new(true);
        updates.status = "Auto-update: ON".to_string();

        Self {
            voice: VoiceManager::new(),
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
        let is_recording = self.voice.is_recording;
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
            .child(
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
                    .child(SharedString::from(self.voice.status.clone()))
                    .child(
                        // Content is kept in sync by the polling task's
                        // `set_value` call above, not here - `Input` has no
                        // per-render value override, unlike some other
                        // gpui-component widgets (e.g. `Clipboard::value`).
                        Input::new(&self.voice_input),
                    ),
            )
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
                    .child(SharedString::from(self.updates.status.clone())),
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
    let mut state = window_handle.lock().unwrap();
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

fn load_icon() -> tray_icon::Icon {
    // The icon is compiled into the binary so it doesn't depend on the
    // (source-tree) path the build ran on; `env!("CARGO_MANIFEST_DIR")`
    // bakes that build machine's path in, which is wrong for installed
    // binaries.
    let image_bytes = include_bytes!("../../cli/assets/icons/prod/icon.png");
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::load_from_memory(image_bytes)
            .expect("Failed to decode embedded icon")
            .into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    tray_icon::Icon::from_rgba(icon_rgba, icon_width, icon_height)
        .expect("Failed to open icon")
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
    gpui_platform::application().run(move |cx| {
        // If a silent update finished while we were running, spawn the
        // helper to apply it as we exit. The subscription must stay alive
        // for the hook to remain registered (same leak trick as the tray
        // icon below). Renaming a running exe is legal on Windows, so the
        // helper's retry loop rides out our shutdown.
        #[cfg(target_os = "windows")]
        std::mem::forget(cx.on_app_quit(|_| async move {
            auto_update::finalize_auto_update_on_quit().await;
        }));
        // Remove stale update/install/old dirs from a crashed previous
        // update, at startup (only empty dirs are removed - a genuinely
        // broken leftover is left alone, see `cleanup_windows`).
        #[cfg(target_os = "windows")]
        cx.spawn(|_: &mut AsyncApp| async move {
            if let Err(error) = auto_update::cleanup_windows().await {
                error!("failed to clean up update dirs: {error}");
            }
        })
        .detach();
        gpui_component::init(cx);
        cx.open_window(
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
        )
        .expect("Failed to create dummy window");
        let main_window_handle: Arc<Mutex<Option<WindowHandle<Root>>>> =
            Arc::new(Mutex::new(None));
        let menu = Menu::new();
        let show_notif_item =
            MenuItem::with_id("show_notif", "Show notification", true, None);
        let exit_item = MenuItem::with_id("exit", "Exit", true, None);
        let _ = menu.append(&show_notif_item);
        let _ = menu.append(&exit_item);
        let icon = load_icon();
        let tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("StealCode")
            .with_icon(icon)
            .with_menu_on_left_click(false)
            .build()
            .expect("Failed to create tray icon");
        std::mem::forget(tray_icon);
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
