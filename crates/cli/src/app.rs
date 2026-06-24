use clap::CommandFactory;
use settings::Settings;
use tracing::{debug, error};

use crate::{
    args::CliArgs,
    commands::{AuthCommands, CliCommands},
};

pub(crate) async fn run_app(
    args: CliArgs,
    settings: &Settings,
) -> anyhow::Result<()> {
    match args.command {
        Some(CliCommands::Acp) => debug!("Start ACP server"),
        Some(CliCommands::Attach { .. }) => debug!("Attach to url"),
        Some(CliCommands::Auth { command }) => match command {
            AuthCommands::List => debug!("List providers and credentials"),
            AuthCommands::Login { .. } => debug!("Login"),
            AuthCommands::Logout => debug!("Logout"),
        },
        Some(CliCommands::Completion { shell }) => {
            let mut cmd = <CliArgs as CommandFactory>::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(
                shell,
                &mut cmd,
                name,
                &mut std::io::stdout(),
            );
        }
        Some(CliCommands::Debug) => debug!("Debug tools"),
        #[cfg(feature = "desktop")]
        Some(CliCommands::Desktop) => {
            if let Some(project) = args.project {
                debug!("Start GUI in {}", project.display());
            } else {
                gui::run_desktop(settings)?;
            }
        }
        Some(CliCommands::Db) => debug!("Database tools"),
        Some(CliCommands::Export { .. }) => debug!("Export session"),
        Some(CliCommands::Github) => debug!("Manage GitHub agent"),
        Some(CliCommands::Import { file }) => debug!("Import from {file}"),
        Some(CliCommands::Mcp) => debug!("Manage MCP servers"),
        Some(CliCommands::Models { .. }) => debug!("List models"),
        Some(CliCommands::Plugin { .. }) => debug!("Install plugin"),
        Some(CliCommands::Pr { .. }) => debug!("Fetch and checkout PR"),
        Some(CliCommands::Providers) => debug!("Manage AI providers"),
        Some(CliCommands::Run { .. }) => debug!("Run with message"),
        Some(CliCommands::Session) => debug!("Manage sessions"),
        #[cfg(feature = "server")]
        Some(CliCommands::Serve) => {
            if let Some(project) = args.project {
                debug!("Start server in {}", project.display());
            } else {
                server::run_server(settings).await?;
            }
        }
        Some(CliCommands::Stats) => debug!("Show stats"),
        Some(CliCommands::Uninstall) => debug!("Uninstall"),
        Some(CliCommands::Upgrade { .. }) => debug!("Upgrade"),
        #[cfg(feature = "web")]
        Some(CliCommands::Web) => {
            if let Some(project) = args.project {
                debug!("Start Web in {}", project.display());
            } else {
                debug!("Start Web");
            }
        }
        #[cfg(feature = "tui")]
        None => {
            let _ = rustls::crypto::ring::default_provider()
                .install_default()
                .inspect_err(|e| {
                    error!("failed to install rustls crypto provider: {e:?}");
                });
            if let Some(project) = args.project {
                debug!("Start TUI in {}", project.display());
            } else {
                tui::run_tui(settings)?;
            }
        }
        #[cfg(not(feature = "tui"))]
        None => {}
    }
    Ok(())
}
