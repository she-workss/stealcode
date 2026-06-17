use anyhow::Context;
use clap::Parser;

use crate::args::CliArgs;
#[cfg(any(feature = "desktop", feature = "server", feature = "web"))]
use crate::commands::CliCommands;

pub(crate) mod app;
pub(crate) mod args;
pub(crate) mod commands;
pub(crate) mod options;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    settings::init_config()?;
    let cwd = std::env::current_dir()
        .context("failed to determine current directory")?;
    let settings = settings::load_settings(&cwd);
    let needs_stdout = match args.command {
        #[cfg(feature = "desktop")]
        Some(CliCommands::Desktop) => false,
        #[cfg(feature = "server")]
        Some(CliCommands::Serve) => false,
        #[cfg(feature = "web")]
        Some(CliCommands::Web) => false,
        None => false,
        _ => args.options.print_logs,
    };
    let _guard = telemetry::init_logging(
        &settings.telemetry.level,
        &settings.telemetry.file_path,
        needs_stdout,
    );
    app::run_app(args, &settings).await?;
    Ok(())
}
