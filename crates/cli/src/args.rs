use std::path::PathBuf;

use clap::{
    Parser,
    builder::{Styles, styling::AnsiColor},
};

use crate::{commands::CliCommands, options::CliOptions};

#[rustfmt::skip]
const HELP_TEMPLATE: &str = "\
stealcode {version}

{usage-heading} {usage}

{all-args}\
";

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().bold())
    .usage(AnsiColor::Green.on_default().bold())
    .literal(AnsiColor::Cyan.on_default().bold())
    .placeholder(AnsiColor::Cyan.on_default());

#[derive(Parser)]
#[command(
    name = "stealcode",
    version = stealcode_version(),
    about = "The open source coding agent.",
    help_template = HELP_TEMPLATE,
    styles = STYLES,
    long_about,
    max_term_width = 80
)]
pub(crate) struct CliArgs {
    /// Path to start stealcode in.
    pub project: Option<PathBuf>,

    #[command(flatten)]
    pub options: CliOptions,

    #[command(subcommand)]
    pub command: Option<CliCommands>,
}

const fn stealcode_version() -> &'static str {
    if let Some(version) = option_env!("STEALCODE_VERSION") {
        version
    } else {
        env!("CARGO_PKG_VERSION")
    }
}
