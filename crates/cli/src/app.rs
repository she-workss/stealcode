use clap::CommandFactory;
use settings::Settings;
use tracing::debug;

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
        Some(CliCommands::Desktop { project }) => {
            #[cfg(target_os = "windows")]
            #[allow(unsafe_code)]
            let _ = unsafe { windows::Win32::System::Console::FreeConsole() };
            if let Some(project) = &project {
                debug!("Start GUI in {}", project.display());
            }
            gui::run_desktop(settings, project.as_deref())?;
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
            if let Some(project) = &args.project {
                debug!("Start server in {}", project.display());
            }
            server::run_server(settings, args.project.as_deref()).await?;
        }
        Some(CliCommands::Stats) => debug!("Show stats"),
        Some(CliCommands::Uninstall) => debug!("Uninstall"),
        Some(CliCommands::Upgrade { target }) => run_upgrade(target).await?,
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
            if let Some(project) = &args.project {
                debug!("Start TUI in {}", project.display());
            }
            tui::run_tui(settings, args.project.as_deref())?;
        }
        #[cfg(not(feature = "tui"))]
        None => {}
    }
    Ok(())
}

/// Implements `stealcode upgrade [target]`. `target`, when given, pins a
/// specific tagged release instead of "latest" (see
/// `commands::CliCommands::Upgrade`).
async fn run_upgrade(target: Option<String>) -> anyhow::Result<()> {
    use anyhow::Context;

    let current_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .context("CARGO_PKG_VERSION is not valid semver")?;
    let channel = release_channel::ReleaseChannel::current();
    let source = auto_update::GithubReleaseSource::new(
        "she-workss",
        "stealcode",
        std::env::var("STEALCODE_GH_TOKEN").ok(),
    );
    let client = reqwest::Client::new();

    let release = match &target {
        Some(version) => {
            auto_update::fetch_release_by_version(&client, &source, version)
                .await?
        }
        None => {
            let Some(release) = auto_update::fetch_release_for_channel(
                &client,
                &source,
                release_channel::ReleaseChannel::current(),
            )
            .await?
            else {
                println!("Релизов пока нет - обновляться нечем.");
                return Ok(());
            };
            release
        }
    };

    let new_version = if target.is_some() {
        // An explicit target version is installed even if it isn't
        // "newer" - the person asked for that exact version.
        Some(release.version()?)
    } else {
        auto_update::newer_version_available(
            &release,
            &current_version,
            channel,
        )?
    };

    let Some(new_version) = new_version else {
        println!("StealCode уже последней версии ({current_version}).");
        return Ok(());
    };

    let (platform, arch) = auto_update::current_platform_arch()?;
    let asset_name = auto_update::expected_asset_name(platform, arch);
    let asset = auto_update::find_asset_by_name(&release, &asset_name)?;
    let download_dir = paths::temp_dir();
    let downloaded_path = download_dir.join(&asset.name);
    auto_update::download_asset(&client, &source, asset, &downloaded_path)
        .await?;

    #[cfg(target_os = "linux")]
    {
        let current_exe = std::env::current_exe()
            .context("failed to determine current executable path")?;
        auto_update::apply_linux_update(&downloaded_path, &current_exe)?;
        println!("Обновлено до {new_version}. Перезапусти StealCode.");
    }

    #[cfg(target_os = "macos")]
    {
        let app_dir = std::env::current_exe()
            .context("failed to determine current executable path")?
            .ancestors()
            .nth(2) // Contents/MacOS/stealcode -> Contents -> StealCode.app
            .context(
                "could not locate StealCode.app from the running executable",
            )?
            .to_path_buf();
        auto_update::apply_macos_update(&downloaded_path, &app_dir)?;
        println!("Обновлено до {new_version}. Перезапусти StealCode.");
    }

    #[cfg(target_os = "windows")]
    {
        let helper_path =
            auto_update::install_release_windows(&downloaded_path).await?;
        anyhow::ensure!(
            helper_path.is_file(),
            "auto_update_helper.exe not found at {} - is StealCode installed via the normal installer?",
            helper_path.display()
        );
        println!(
            "Обновление до {new_version} подготовлено. Перезапусти StealCode, чтобы применить."
        );
    }

    Ok(())
}
