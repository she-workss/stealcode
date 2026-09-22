//! Mobile entry point for `StealCode` (`libmobile.so` on Android,
//! `libmobile.a` on iOS).
//!
//! The UI (root view + window creation) is shared between both platforms;
//! only the platform initialisation differs.

#[cfg(any(target_os = "android", target_os = "ios"))]
use gpui::{
    App, Context, Entity, IntoElement, Render, Window, div, prelude::*,
};
#[cfg(any(target_os = "android", target_os = "ios"))]
use gpui_kit::{
    component::{button::*, *},
    *,
};

/// Minimal mobile root view.
#[cfg(any(target_os = "android", target_os = "ios"))]
#[derive(Debug)]
struct MobileApp;

#[cfg(any(target_os = "android", target_os = "ios"))]
impl Render for MobileApp {
    fn render(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        div()
            .v_flex()
            .gap_2()
            .size_full()
            .items_center()
            .justify_center()
            .child("Hello, World!")
            .child(Button::new("ok").primary().label("Let's Go!"))
    }
}

#[cfg(any(target_os = "android", target_os = "ios"))]
fn mobile_root(window: &mut Window, cx: &mut App) -> Entity<Root> {
    let view = cx.new(|_| MobileApp);
    cx.new(|cx| Root::new(view, window, cx))
}

/// Open the fullscreen main window. Shared by the Android and iOS entry points.
#[cfg(any(target_os = "android", target_os = "ios"))]
fn open_main_window(cx: &mut App) {
    gpui_kit::init(cx);
    if let Err(e) = cx.open_window(
        gpui::WindowOptions {
            window_bounds: None,
            ..Default::default()
        },
        mobile_root,
    ) {
        log::error!("cx.open_window failed: {e:#}");
    }

    cx.activate(true);
}

/// Called by the `android-activity` crate on a dedicated native thread.
/// Does NOT return until the app is ready to exit.
#[cfg(target_os = "android")]
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
fn android_main(app: android_activity::AndroidApp) {
    use gpui_mobile::android::jni;

    // Logger first - so everything after this is visible in logcat.
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("mobile"),
    );

    // Panic hook - routes panics to logcat instead of silently aborting.
    jni::install_panic_hook();

    log::info!("android_main: entered");

    // Initialise the global AndroidApp + AndroidPlatform.
    let _platform = jni::init_platform(&app);
    log::info!("android_main: platform initialised");

    // Get a SharedPlatform (Rc-compatible wrapper around the global
    // Arc<AndroidPlatform>) so we can hand it to GPUI.
    let shared = match jni::shared_platform() {
        Some(s) => s,
        None => {
            log::error!(
                "android_main: shared_platform() returned None - aborting"
            );
            return;
        }
    };

    log::info!("android_main: creating GPUI Application");

    // `Application::with_platform(...).run(...)` calls `Platform::run` which,
    // on Android, blocks by driving the native event loop. The `|cx| { ... }`
    // closure runs once `MainEvent::InitWindow` has delivered a native
    // surface, keeping the `Application` alive for the loop's lifetime.
    gpui::Application::with_platform(shared.into_rc()).run(open_main_window);

    log::info!("android_main: Application.run returned - activity will finish");
}

/// Minimal logger that routes Rust `log` crate messages through NSLog.
#[cfg(target_os = "ios")]
struct NsLogLogger;

#[cfg(target_os = "ios")]
impl log::Log for NsLogLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let msg = format!(
                "[{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
            nslog(&msg);
        }
    }

    fn flush(&self) {}
}

/// Call NSLog from Rust via raw FFI.
#[cfg(target_os = "ios")]
#[allow(unsafe_code)]
fn nslog(msg: &str) {
    use objc2::{class, msg_send, runtime::AnyObject};
    unsafe {
        unsafe extern "C" {
            fn NSLog(fmt: *mut AnyObject, ...);
        }
        let c_msg = std::ffi::CString::new(msg).unwrap_or_default();
        let ns_msg: *mut AnyObject = msg_send![class!(NSString), alloc];
        let ns_msg: *mut AnyObject =
            msg_send![ns_msg, initWithUTF8String: c_msg.as_ptr()];
        let c_fmt = std::ffi::CString::new("%@").unwrap_or_default();
        let ns_fmt: *mut AnyObject = msg_send![class!(NSString), alloc];
        let ns_fmt: *mut AnyObject =
            msg_send![ns_fmt, initWithUTF8String: c_fmt.as_ptr()];
        NSLog(ns_fmt, ns_msg);
    }
}

/// Register the StealCode root view with the GPUI iOS platform.
///
/// Called from `App.swift` before `gpui_ios_run_demo()`. The symbol lives in
/// this crate's static lib, which Xcode force-loads.
#[cfg(target_os = "ios")]
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn gpui_ios_register_app() {
    // Set up Rust logging -> NSLog so log::info! etc. appear in devicectl
    // --console.
    let _ = log::set_logger(&NsLogLogger)
        .map(|()| log::set_max_level(log::LevelFilter::Info));

    // Panic hook -> NSLog so panics are visible.
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("GPUI PANIC: {info}");
        nslog(&msg);
    }));

    gpui_mobile::ios::ffi::set_app_callback(Box::new(open_main_window));
}

/// Convenience entry point for a binary target (not used by `App.swift`).
#[cfg(target_os = "ios")]
pub fn ios_main() {
    gpui_ios_register_app();
    gpui_mobile::ios::ffi::run_app();
}
