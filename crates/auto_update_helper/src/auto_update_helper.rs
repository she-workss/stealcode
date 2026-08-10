#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(windows)]
mod dialog;
mod updater;

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
        "auto_update_helper only does anything on Windows; macOS and Linux \
         apply updates directly (see auto_update::apply_linux_update / apply_macos_update)"
    );
    std::process::exit(1);
}

#[cfg(windows)]
mod windows_impl {
    use std::borrow::Cow;

    use anyhow::{Context, Result};
    use tracing::info;
    use windows::Win32::UI::WindowsAndMessaging::{
        DispatchMessageW, GetMessageW, MB_ICONERROR, MB_SYSTEMMODAL, MSG,
        MessageBoxW,
    };

    use crate::{dialog, updater};

    #[derive(Debug, Default)]
    struct Args {
        launch: bool,
    }

    pub(crate) fn run() -> Result<()> {
        let helper_dir = std::env::current_exe()?
            .parent()
            .context("no parent dir")?
            .to_path_buf();
        let app_dir =
            helper_dir.parent().context("no parent dir")?.to_path_buf();
        info!("Starting StealCode update");
        let args = parse_args(std::env::args().skip(1));
        let hwnd =
            dialog::create_dialog_window(updater::jobs().len())?.0 as isize;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = updater::perform_update(&app_dir, args.launch);
            tx.send(result).ok();
            dialog::notify_terminate(hwnd);
        });
        unsafe {
            let mut message = MSG::default();
            while GetMessageW(&mut message, None, 0, 0).as_bool() {
                DispatchMessageW(&message);
            }
        }
        if let Ok(Err(error)) = rx.try_recv() {
            return Err(error);
        }
        Ok(())
    }

    fn parse_args(input: impl IntoIterator<Item = String>) -> Args {
        let mut args = Args { launch: true };
        let mut input = input.into_iter();
        if let Some(arg) = input.next() {
            let launch_arg = if arg == "--launch" {
                input.next().map(Cow::Owned)
            } else {
                arg.strip_prefix("--launch=")
                    .map(|s| Cow::Owned(s.to_string()))
            };
            if launch_arg.as_deref() == Some("false") {
                args.launch = false;
            }
        }
        args
    }

    pub(crate) fn show_error(mut content: String) {
        if content.len() > 600 {
            content.truncate(600);
            content.push_str("...");
        }
        let _ = unsafe {
            MessageBoxW(
                None,
                &windows::core::HSTRING::from(content),
                windows::core::w!("StealCode update failed"),
                MB_ICONERROR | MB_SYSTEMMODAL,
            )
        };
    }

    #[cfg(test)]
    mod tests {
        use super::parse_args;

        #[test]
        fn launch_defaults_to_true() {
            assert!(parse_args([]).launch);
        }

        #[test]
        fn launch_false_via_two_args() {
            assert!(!parse_args(["--launch".into(), "false".into()]).launch);
        }

        #[test]
        fn launch_false_via_one_arg() {
            assert!(!parse_args(["--launch=false".into()]).launch);
        }

        #[test]
        fn launch_true_is_the_default_for_any_other_value() {
            assert!(parse_args(["--launch".into(), "true".into()]).launch);
            assert!(parse_args(["--launch".into(), "yes".into()]).launch);
        }
    }
}
