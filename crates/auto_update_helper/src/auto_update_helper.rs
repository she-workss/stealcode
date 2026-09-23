#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(windows)]
mod dialog;
#[cfg(any(windows, test))]
mod updater;
#[cfg(windows)]
mod windows_impl;

#[cfg(windows)]
fn main() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .try_init();
    if let Err(error) = windows_impl::run() {
        tracing::error!("StealCode update failed: {error:?}");
        windows_impl::show_error(format!("{error:?}"));
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!(
        "auto-update-helper only does anything on Windows; macOS and Linux \
         apply updates directly (see auto_update::apply_linux_update / \
         apply_macos_update)"
    );
    std::process::exit(1);
}
