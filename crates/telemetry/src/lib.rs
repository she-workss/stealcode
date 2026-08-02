//! Telemetry for StealCode: file- and stdout-based `tracing` logging setup.

use std::{fs::OpenOptions, io, path::Path};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::{
    EnvFilter, Registry,
    fmt::{self, format::FmtSpan, time::ChronoLocal},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

fn setup_log_file(file: impl AsRef<Path>) -> (NonBlocking, WorkerGuard) {
    let log_dir = paths::logs_dir();
    if let Err(e) = std::fs::create_dir_all(log_dir) {
        eprintln!(
            "Warning: failed to create log directory {}: {e}",
            log_dir.display()
        );
    }
    let log_path = log_dir.join(file);
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to open log file {}: {e}", log_path.display());
            panic!("Could not open log file: {e}");
        });
    tracing_appender::non_blocking(log_file)
}

fn build_env_filter(level_filter: &str) -> EnvFilter {
    EnvFilter::try_new(level_filter).unwrap_or_else(|_| EnvFilter::new("info"))
}

fn custom_timer() -> ChronoLocal {
    ChronoLocal::new("%d.%m.%Y %T".to_string())
}

pub fn init_logging(
    level_filter: impl AsRef<str>,
    file_name: impl AsRef<Path>,
    to_stdout: bool,
) -> Option<WorkerGuard> {
    if tracing::dispatcher::has_been_set() {
        return None;
    }
    let (non_blocking_appender, guard) = setup_log_file(file_name.as_ref());
    let filter = build_env_filter(level_filter.as_ref());
    let file_layer = fmt::layer()
        .with_writer(non_blocking_appender)
        .with_timer(custom_timer())
        .with_ansi(false)
        .with_span_events(FmtSpan::CLOSE);
    let stdout_layer = to_stdout.then(|| {
        fmt::layer()
            .with_writer(io::stdout)
            .with_target(false)
            .with_timer(custom_timer())
            .with_ansi(true)
            .with_span_events(FmtSpan::CLOSE)
    });
    Registry::default()
        .with(filter)
        .with(file_layer)
        .with(stdout_layer)
        .init();
    Some(guard)
}
