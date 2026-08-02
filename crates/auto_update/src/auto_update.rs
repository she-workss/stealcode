//! StealCode's self-updater, structured like Zed's `auto_update` crate
//! (single-file `src/auto_update.rs`, paired with a separate
//! `auto_update_helper` binary for the Windows-only file swap).
//!
//! This is an independent implementation written for StealCode, not a copy
//! of Zed's GPL-3.0-licensed `crates/auto_update` - the architecture is the
//! same idea (which isn't copyrightable), the code is StealCode's own, kept
//! under the workspace's MIT license.
//!
//! Differences from Zed worth knowing about:
//! - StealCode ships one executable, no COM shell extension DLL, no appx
//!   package - so `auto_update_helper` doesn't need Windows Restart Manager.
//!   Nothing but our own process ever holds a lock on `stealcode.exe`.
//! - On Linux, Zed distributes a whole `zed.app/` folder under `~/.local` and
//!   `rsync`s over it. StealCode ships a single binary that's just somewhere on
//!   PATH, so a plain `tar.gz` + atomic rename is enough - there's no folder to
//!   mirror.
//! - Release channel awareness (stable vs nightly) lives in the separate
//!   `release_channel` crate, matching Zed's split.

use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use release_channel::ReleaseChannel;
use reqwest::header::{ACCEPT, USER_AGENT};
use semver::Version;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

const USER_AGENT_VALUE: &str = "stealcode-auto-update";

// ---------------------------------------------------------------------
// Settings: whether background polling is enabled at all. Manual checks
// always work regardless of this. See `settings::Settings::auto_update`.
// ---------------------------------------------------------------------

/// Whether an update check was triggered by background polling or by the
/// person explicitly asking ("Check for updates now"). Mirrors Zed's
/// `UpdateCheckType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateCheckType {
    Automatic,
    Manual,
}

impl UpdateCheckType {
    #[must_use]
    pub const fn is_manual(self) -> bool {
        matches!(self, Self::Manual)
    }
}

/// Reads the `auto_update` setting (default `true`), matching Zed's
/// `content.auto_update: Option<bool>`.
#[must_use]
pub fn auto_update_setting_enabled(settings: &settings::Settings) -> bool {
    settings.auto_update.unwrap_or(true)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateStatus {
    Idle,
    Checking,
    UpToDate,
    Downloading {
        version: Version,
    },
    Installing {
        version: Version,
    },
    /// Windows only: staged in the background, waiting for the app to quit
    /// (or an explicit restart) before the swap actually happens.
    PendingRestart {
        version: Version,
    },
    Updated {
        version: Version,
    },
    Errored {
        message: String,
    },
}

// ---------------------------------------------------------------------
// GitHub release metadata + asset matching
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ReleaseAsset {
    pub id: u64,
    pub name: String,
    pub size: u64,
    /// The GitHub API URL for this asset, e.g.
    /// `https://api.github.com/repos/{owner}/{repo}/releases/assets/{id}`.
    /// Required (with an `Accept: application/octet-stream` header and a
    /// bearer token) to download assets from a private repository.
    pub url: String,
    /// The public download URL. Only resolves without authentication once
    /// the repository is public.
    pub browser_download_url: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ReleaseInfo {
    pub tag_name: String,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub prerelease: bool,
    pub assets: Vec<ReleaseAsset>,
}

impl ReleaseInfo {
    /// Parses the semantic version out of `tag_name`, tolerating a leading
    /// `v` (e.g. `v0.3.1` -> `0.3.1`).
    pub fn version(&self) -> Result<Version> {
        let raw = self.tag_name.trim_start_matches('v');
        Version::parse(raw).with_context(|| {
            format!(
                "release tag {:?} is not a valid semver version",
                self.tag_name
            )
        })
    }
}

pub fn parse_release_response(body: &[u8]) -> Result<ReleaseInfo> {
    serde_json::from_slice(body).context("failed to parse GitHub release JSON")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Linux,
    MacOs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    const fn as_str(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
}

pub fn current_platform_arch() -> Result<(Platform, Arch)> {
    platform_arch_for(env::consts::OS, env::consts::ARCH)
}

fn platform_arch_for(os: &str, arch: &str) -> Result<(Platform, Arch)> {
    let platform = match os {
        "windows" => Platform::Windows,
        "linux" => Platform::Linux,
        "macos" => Platform::MacOs,
        other => anyhow::bail!("self-update is not supported on OS {other:?}"),
    };
    let arch = match arch {
        "x86_64" => Arch::X86_64,
        "aarch64" => Arch::Aarch64,
        other => anyhow::bail!(
            "self-update is not supported on architecture {other:?}"
        ),
    };
    Ok((platform, arch))
}

/// `StealCode-{arch}.exe` / `StealCode-{arch}.dmg` /
/// `stealcode-linux-{arch}.tar.gz`, matching Zed's naming. Keep in sync with
/// `.github/workflows/release.yml`.
#[must_use]
pub fn expected_asset_name(platform: Platform, arch: Arch) -> String {
    match platform {
        Platform::Windows => format!("StealCode-{}.exe", arch.as_str()),
        Platform::MacOs => format!("StealCode-{}.dmg", arch.as_str()),
        Platform::Linux => format!("stealcode-linux-{}.tar.gz", arch.as_str()),
    }
}

pub fn find_asset_by_name<'a>(
    release: &'a ReleaseInfo,
    name: &str,
) -> Result<&'a ReleaseAsset> {
    release
        .assets
        .iter()
        .find(|asset| asset.name == name)
        .with_context(|| {
            format!(
                "release {:?} has no asset named {name:?} (available: {:?})",
                release.tag_name,
                release.assets.iter().map(|a| &a.name).collect::<Vec<_>>()
            )
        })
}

pub fn is_update_available(current: &Version, candidate: &Version) -> bool {
    candidate > current
}

/// Channel-aware: nightly accepts GitHub prereleases, stable never does.
pub fn newer_version_available(
    release: &ReleaseInfo,
    current: &Version,
    channel: ReleaseChannel,
) -> Result<Option<Version>> {
    if release.draft {
        return Ok(None);
    }
    if release.prerelease && !channel.accepts_prereleases() {
        return Ok(None);
    }
    let candidate = release.version()?;
    Ok(is_update_available(current, &candidate).then_some(candidate))
}

// ---------------------------------------------------------------------
// GitHub networking
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GithubReleaseSource {
    pub owner: String,
    pub repo: String,
    /// A fine-grained personal access token scoped to this repository with
    /// `Contents: Read-only` permission. Only needed while the repository is
    /// private; must never be embedded in a binary distributed to others.
    pub token: Option<String>,
    api_base: String,
}

impl GithubReleaseSource {
    pub fn new(
        owner: impl Into<String>,
        repo: impl Into<String>,
        token: Option<String>,
    ) -> Self {
        Self {
            owner: owner.into(),
            repo: repo.into(),
            token,
            api_base: "https://api.github.com".to_string(),
        }
    }

    #[cfg(test)]
    fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into();
        self
    }

    fn latest_release_url(&self) -> String {
        format!(
            "{}/repos/{}/{}/releases/latest",
            self.api_base, self.owner, self.repo
        )
    }

    fn release_by_tag_url(&self, tag: &str) -> String {
        format!(
            "{}/repos/{}/{}/releases/tags/{}",
            self.api_base, self.owner, self.repo, tag
        )
    }

    fn releases_list_url(&self) -> String {
        format!(
            "{}/repos/{}/{}/releases",
            self.api_base, self.owner, self.repo
        )
    }
}

/// Parses a `GET /repos/{owner}/{repo}/releases` response body (an array,
/// not a single object like `/releases/latest`).
pub fn parse_releases_list_response(body: &[u8]) -> Result<Vec<ReleaseInfo>> {
    serde_json::from_slice(body)
        .context("failed to parse GitHub releases list JSON")
}

/// Fetches the single most recent, non-draft release regardless of
/// prerelease status - needed for the nightly channel, since
/// `GET /releases/latest` explicitly skips prereleases and would never
/// surface a nightly build marked `prerelease: true` on GitHub. Returns
/// `Ok(None)` when the repository has no non-draft releases yet.
pub async fn fetch_most_recent_release(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
) -> Result<Option<ReleaseInfo>> {
    let mut request = client
        .get(source.releases_list_url())
        .header(USER_AGENT, USER_AGENT_VALUE)
        .header(ACCEPT, "application/vnd.github+json");
    if let Some(token) = &source.token {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .context("failed to reach the GitHub releases API")?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .context("failed to read the releases list body")?;
    anyhow::ensure!(
        status.is_success(),
        "GitHub releases API returned {status}: {}",
        String::from_utf8_lossy(&body)
    );

    let releases = parse_releases_list_response(&body)?;
    Ok(releases.into_iter().find(|release| !release.draft))
}

/// Dispatches to the right fetch strategy for the given channel: stable
/// uses `/releases/latest` (which correctly ignores prereleases), nightly
/// uses the full list (since it needs to see prerelease-flagged builds).
/// Returns `Ok(None)` when no release exists for this channel yet.
pub async fn fetch_release_for_channel(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
    channel: ReleaseChannel,
) -> Result<Option<ReleaseInfo>> {
    match channel {
        ReleaseChannel::Stable => fetch_latest_release(client, source).await,
        ReleaseChannel::Nightly => {
            fetch_most_recent_release(client, source).await
        }
    }
}

/// `GET /releases/latest` returns 404 both when the repository has no
/// releases at all and when every release is a draft or prerelease - the
/// GitHub API's documented behavior. Treat that as "no update available"
/// (`Ok(None)`) rather than an error.
pub async fn fetch_latest_release(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
) -> Result<Option<ReleaseInfo>> {
    fetch_release(client, source, &source.latest_release_url()).await
}

/// Fetches a specific tagged release, for `stealcode upgrade <version>`
/// (the `target` field on `CliCommands::Upgrade`). `version` may be given
/// with or without a leading `v`. Unlike `fetch_latest_release`, a missing
/// release here is an error - the person asked for that exact version.
pub async fn fetch_release_by_version(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
    version: &str,
) -> Result<ReleaseInfo> {
    let tag = if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{version}")
    };
    fetch_release(client, source, &source.release_by_tag_url(&tag))
        .await?
        .with_context(|| {
            format!(
                "release {tag:?} not found - either the tag does not exist, \
                 or the repository is private and no STEALCODE_GH_TOKEN was provided"
            )
        })
}

async fn fetch_release(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
    url: &str,
) -> Result<Option<ReleaseInfo>> {
    let mut request = client
        .get(url)
        .header(USER_AGENT, USER_AGENT_VALUE)
        .header(ACCEPT, "application/vnd.github+json");
    if let Some(token) = &source.token {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .context("failed to reach the GitHub releases API")?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .context("failed to read the release response body")?;
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    anyhow::ensure!(
        status.is_success(),
        "GitHub releases API returned {status}: {}",
        String::from_utf8_lossy(&body)
    );
    parse_release_response(&body).map(Some)
}

pub async fn download_asset(
    client: &reqwest::Client,
    source: &GithubReleaseSource,
    asset: &ReleaseAsset,
    destination: &Path,
) -> Result<()> {
    let mut request = if source.token.is_some() {
        // Private repo: must go through the API asset endpoint with this
        // exact Accept header. The API responds with either a 200
        // (streamed directly) or a 302 redirect to a pre-signed storage
        // URL; reqwest follows redirects by default, so both cases are
        // handled transparently.
        client
            .get(&asset.url)
            .header(ACCEPT, "application/octet-stream")
    } else {
        // Public repo: the direct download URL works without auth.
        client.get(&asset.browser_download_url)
    };
    request = request.header(USER_AGENT, USER_AGENT_VALUE);
    if let Some(token) = &source.token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await.with_context(|| {
        format!("failed to start downloading asset {:?}", asset.name)
    })?;
    let status = response.status();
    anyhow::ensure!(
        status.is_success(),
        "failed to download asset {:?}: HTTP {status}",
        asset.name
    );

    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!("failed to create directory {}", parent.display())
        })?;
    }
    let mut file =
        tokio::fs::File::create(destination)
            .await
            .with_context(|| {
                format!("failed to create {}", destination.display())
            })?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.context("error while streaming the asset download")?;
        file.write_all(&chunk)
            .await
            .context("failed to write a downloaded chunk to disk")?;
    }
    file.flush()
        .await
        .context("failed to flush the downloaded file")?;
    Ok(())
}

/// Convenience synchronous entry point for callers that aren't already
/// inside a tokio runtime - the TUI's blocking event loop, or a GPUI
/// button's `on_click` handler running on a plain OS thread. Spins up its
/// own single-threaded runtime for the duration of the check, so it can be
/// called from anywhere.
pub fn check_now_blocking(
    owner: &str,
    repo: &str,
    token: Option<String>,
    current_version: Version,
    channel: ReleaseChannel,
) -> Result<Option<Version>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start update-check runtime")?;
    runtime.block_on(async move {
        let source = GithubReleaseSource::new(owner, repo, token);
        let client = reqwest::Client::new();
        match fetch_latest_release(&client, &source).await? {
            Some(release) => {
                newer_version_available(&release, &current_version, channel)
            }
            None => Ok(None),
        }
    })
}

/// Blocking "download and install now" for the same non-tokio callers as
/// `check_now_blocking` (the GUI's background worker, the TUI's event
/// loop). Runs the full update flow: fetch the latest release for the
/// channel, download the matching asset, then stage it per platform - on
/// Windows the silent installer writes into `install\` (the swap happens
/// later via the helper), on Linux/macOS the new binary is applied in
/// place. Returns the new version on success. Does not restart the app -
/// call `restart_updated_app` afterwards.
pub fn update_now_blocking(
    owner: &str,
    repo: &str,
    token: Option<String>,
    current_version: Version,
    channel: ReleaseChannel,
) -> Result<Version> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start update runtime")?;
    runtime.block_on(async move {
        let source = GithubReleaseSource::new(owner, repo, token);
        let client = reqwest::Client::new();
        let release = fetch_release_for_channel(&client, &source, channel)
            .await?
            .context("no release available for this channel")?;
        let new_version = newer_version_available(
            &release,
            &current_version,
            channel,
        )?
        .context("no update available")?;

        let (platform, arch) = current_platform_arch()?;
        let asset_name = expected_asset_name(platform, arch);
        let asset = find_asset_by_name(&release, &asset_name)?;
        let download_dir = std::env::temp_dir().join("stealcode");
        let downloaded_path = download_dir.join(&asset.name);
        download_asset(&client, &source, asset, &downloaded_path).await?;

        #[cfg(target_os = "linux")]
        {
            let current_exe = std::env::current_exe()
                .context("failed to determine current executable path")?;
            apply_linux_update(&downloaded_path, &current_exe)?;
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
            apply_macos_update(&downloaded_path, &app_dir)?;
        }

        #[cfg(target_os = "windows")]
        {
            let helper_path = install_release_windows(&downloaded_path).await?;
            anyhow::ensure!(
                helper_path.is_file(),
                "auto_update_helper.exe not found at {} - is StealCode installed via the normal installer?",
                helper_path.display()
            );
        }

        Ok(new_version)
    })
}

/// Relaunches StealCode after a successful `update_now_blocking` and exits
/// the current process. On Windows the binary swap is done by
/// `auto_update_helper.exe --launch true` (see `restart_and_update`); on
/// Unix the new binary is already in place, so the current executable is
/// simply re-executed. Never returns on success.
pub fn restart_updated_app() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        restart_and_update()
    }
    #[cfg(unix)]
    {
        let exe = std::env::current_exe()?;
        std::process::Command::new(&exe)
            .spawn()
            .context("failed to relaunch StealCode")?;
        std::process::exit(0);
    }
}

// ---------------------------------------------------------------------
// Linux / macOS: no helper process needed - rename() over a running exe is
// allowed. macOS mounts the dmg via `hdiutil`; Linux extracts the tar.gz.
// ---------------------------------------------------------------------

pub fn atomic_swap(new_binary: &Path, target_path: &Path) -> Result<()> {
    anyhow::ensure!(
        new_binary.parent() == target_path.parent(),
        "refusing a cross-filesystem swap from {} to {}",
        new_binary.display(),
        target_path.display()
    );
    std::fs::rename(new_binary, target_path).with_context(|| {
        format!(
            "failed to swap {} into {}",
            new_binary.display(),
            target_path.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(target_path)?.permissions();
        perms.set_mode(perms.mode() | 0o755);
        std::fs::set_permissions(target_path, perms)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn apply_linux_update(tarball: &Path, current_exe: &Path) -> Result<()> {
    let extract_dir = tempdir_next_to(current_exe)?;
    let output = std::process::Command::new("tar")
        .arg("-xzf")
        .arg(tarball)
        .arg("-C")
        .arg(&extract_dir)
        .output()
        .context("failed to spawn `tar`")?;
    anyhow::ensure!(
        output.status.success(),
        "tar extraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let extracted_binary = extract_dir.join("stealcode");
    anyhow::ensure!(
        extracted_binary.is_file(),
        "archive did not contain a `stealcode` binary"
    );

    let staged_path = current_exe
        .parent()
        .context("current executable has no parent directory")?
        .join(".stealcode.update.tmp");
    std::fs::rename(&extracted_binary, &staged_path)?;
    std::fs::remove_dir_all(&extract_dir).ok();
    atomic_swap(&staged_path, current_exe)
}

#[cfg(target_os = "macos")]
pub fn apply_macos_update(
    dmg_path: &Path,
    installed_app_dir: &Path,
) -> Result<()> {
    let mount_point = tempdir_next_to(installed_app_dir)
        .unwrap_or_else(|_| std::env::temp_dir().join("stealcode-dmg-mount"));
    std::fs::create_dir_all(&mount_point)?;

    let attach = std::process::Command::new("hdiutil")
        .arg("attach")
        .arg(dmg_path)
        .arg("-mountpoint")
        .arg(&mount_point)
        .arg("-nobrowse")
        .arg("-quiet")
        .output()
        .context("failed to spawn `hdiutil attach`")?;
    anyhow::ensure!(
        attach.status.success(),
        "hdiutil attach failed: {}",
        String::from_utf8_lossy(&attach.stderr)
    );

    let result = (|| -> Result<()> {
        let source_app = mount_point.join("StealCode.app");
        anyhow::ensure!(
            source_app.is_dir(),
            "dmg did not contain StealCode.app"
        );
        let rsync = std::process::Command::new("rsync")
            .arg("-a")
            .arg("--delete")
            .arg(format!("{}/", source_app.display()))
            .arg(format!("{}/", installed_app_dir.display()))
            .output()
            .context("failed to spawn `rsync`")?;
        anyhow::ensure!(
            rsync.status.success(),
            "rsync failed: {}",
            String::from_utf8_lossy(&rsync.stderr)
        );
        Ok(())
    })();

    let _ = std::process::Command::new("hdiutil")
        .arg("detach")
        .arg(&mount_point)
        .arg("-quiet")
        .output();
    std::fs::remove_dir_all(&mount_point).ok();
    result
}

#[cfg(unix)]
fn tempdir_next_to(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().context("path has no parent directory")?;
    let unique = format!(
        ".stealcode-update-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let dir = parent.join(unique);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

// ---------------------------------------------------------------------
// Windows: stage the installer into a shadow `install\` directory while
// StealCode keeps running, in the background - same trick Zed uses, minus
// Restart Manager (nothing locks our files except our own process, since
// StealCode has no context-menu DLL).
// ---------------------------------------------------------------------

/// Runs the downloaded installer silently with `/update=true`, which
/// `stealcode.iss` interprets as "install into `{app}\install\` instead of
/// `{app}` directly, and write `updates\versions.txt` when done" - see the
/// `IsUpdating`/`GetInstallDir` Pascal functions there. Returns the path to
/// `auto_update_helper.exe`, which the caller should invoke later (at quit
/// time, or immediately for an explicit restart).
#[cfg(windows)]
pub async fn install_release_windows(
    downloaded_installer: &Path,
) -> Result<PathBuf> {
    let output = tokio::process::Command::new(downloaded_installer)
        .arg("/verysilent")
        .arg("/update=true")
        .arg("/MERGETASKS=!desktopicon")
        .output()
        .await
        .context("failed to run the installer silently")?;
    anyhow::ensure!(
        output.status.success(),
        "installer exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let helper_path = std::env::current_exe()?
        .parent()
        .context("no parent dir for stealcode.exe")?
        .join("tools")
        .join("auto_update_helper.exe");
    Ok(helper_path)
}

/// Removes stale `updates\`/`install\`/`old\` directories left over from a
/// crashed or interrupted update, at startup. `remove_dir_all` is used but
/// every error is swallowed: a leftover `old\stealcode.exe` may still be the
/// mapped image of a concurrently running instance, in which case removing it
/// fails and it is simply rolled into next time.
#[cfg(windows)]
pub async fn cleanup_windows() -> Result<()> {
    let parent = std::env::current_exe()?
        .parent()
        .context("no parent dir for stealcode.exe")?
        .to_owned();
    let _ = tokio::fs::remove_dir_all(parent.join("updates")).await;
    let _ = tokio::fs::remove_dir_all(parent.join("install")).await;
    let _ = tokio::fs::remove_dir_all(parent.join("old")).await;
    Ok(())
}

/// Applies a staged update (one written by an earlier session via
/// `install_release_windows`) when the current process is started *after*
/// that staging finished. Used by the GUI and the TUI at startup: if
/// `install\stealcode.exe` and `updates\versions.txt` both exist, the helper is
/// spawned with `--launch true` so the swap happens and StealCode relaunches
/// as the new version; the caller should then exit the current (stale)
/// process. Returns true when the swap was handed off.
#[cfg(windows)]
pub fn apply_staged_update_on_startup() -> Result<bool> {
    let exe = std::env::current_exe()?;
    let app_dir = exe
        .parent()
        .context("no parent dir for stealcode.exe")?
        .to_owned();
    let flag_file = app_dir.join("updates").join("versions.txt");
    let staged = app_dir.join("install").join("stealcode.exe");
    if !flag_file.exists() || !staged.is_file() {
        return Ok(false);
    }
    let helper = app_dir.join("tools").join("auto_update_helper.exe");
    anyhow::ensure!(
        helper.is_file(),
        "auto_update_helper.exe not found at {} - is StealCode installed via the normal installer?",
        helper.display()
    );
    std::process::Command::new(&helper)
        .arg("--launch")
        .arg("true")
        .status()
        .context("failed to spawn auto_update_helper.exe")?;
    Ok(true)
}

/// Blocking variant of `apply_staged_update_on_startup` for the sync hosts
/// (TUI event loop, GUI worker) that don't pull in tokio themselves.
#[cfg(windows)]
pub fn apply_staged_update_on_startup_blocking() -> Result<bool> {
    block_on_sync_host(apply_staged_update_on_startup())
}

/// Called from the app's quit hook. If the background silent install
/// finished (marked by `updates\versions.txt`, written by the installer's
/// `[Code]` section - see `stealcode.iss`), spawns the helper to do the
/// actual file swap. If it hasn't finished yet, does nothing - the update
/// simply isn't applied this session and will be picked up next time.
#[cfg(windows)]
pub async fn finalize_auto_update_on_quit() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(app_dir) = exe.parent() else { return };
    let flag_file = app_dir.join("updates").join("versions.txt");
    if !flag_file.exists() {
        return;
    }
    let helper = app_dir.join("tools").join("auto_update_helper.exe");
    if let Ok(mut child) = tokio::process::Command::new(helper)
        .arg("--launch")
        .arg("false")
        .spawn()
    {
        let _ = child.wait().await;
    }
}

/// Runs a future to completion on a dedicated thread with its own
/// single-threaded tokio runtime. The sync hosts that call the `_blocking`
/// wrappers may themselves be running inside another tokio runtime (the
/// CLI's `#[tokio::main]` drives the TUI), where creating a runtime or
/// `block_on`-ing on the current thread panics with "Cannot start a
/// runtime from within a runtime" - a fresh thread has no such context.
#[cfg(target_os = "windows")]
fn block_on_sync_host<F, R>(future: F) -> R
where
    F: std::future::Future<Output = R> + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to start update runtime thread");
        let result = runtime.block_on(future);
        let _ = tx.send(result);
    });
    rx.recv().expect("update runtime thread panicked")
}

/// Blocking variant of `finalize_auto_update_on_quit` for hosts with a
/// blocking event loop (the TUI) that don't pull in tokio themselves.
#[cfg(target_os = "windows")]
pub fn finalize_auto_update_on_quit_blocking() {
    block_on_sync_host(finalize_auto_update_on_quit());
}

/// Blocking variant of `cleanup_windows` for the same sync hosts.
#[cfg(target_os = "windows")]
pub fn cleanup_windows_blocking() -> Result<()> {
    block_on_sync_host(cleanup_windows())
}

/// Called when the person explicitly clicks "Restart to update" while
/// StealCode is running: relaunches via the helper (which then launches the
/// new binary itself), instead of quietly deferring to the next quit. Never
/// returns on success - the process must exit for the swap to happen.
#[cfg(windows)]
pub fn restart_and_update() -> Result<()> {
    let exe = std::env::current_exe()?;
    let app_dir = exe.parent().context("no parent dir for stealcode.exe")?;
    let helper = app_dir.join("tools").join("auto_update_helper.exe");
    std::process::Command::new(helper)
        .arg("--launch")
        .arg("true")
        .spawn()
        .context("failed to spawn auto_update_helper.exe")?;
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_release_json(
        tag: &str,
        prerelease: bool,
        assets: &[(&str, &str)],
    ) -> String {
        let assets_json: Vec<String> = assets
            .iter()
            .enumerate()
            .map(|(i, (name, url))| {
                format!(
                    r#"{{"id": {id}, "name": "{name}", "size": 1, "url": "https://api.github.com/repos/she-workss/stealcode/releases/assets/{id}", "browser_download_url": "{url}"}}"#,
                    id = i + 1
                )
            })
            .collect();
        format!(
            r#"{{"tag_name": "{tag}", "draft": false, "prerelease": {prerelease}, "assets": [{}]}}"#,
            assets_json.join(",")
        )
    }

    #[test]
    fn parses_a_realistic_release_response() {
        let body = sample_release_json(
            "v0.3.1",
            false,
            &[("StealCode-x86_64.exe", "https://x/StealCode-x86_64.exe")],
        );
        let release = parse_release_response(body.as_bytes()).unwrap();
        assert_eq!(release.version().unwrap(), Version::new(0, 3, 1));
        assert_eq!(release.assets.len(), 1);
    }

    #[test]
    fn expected_asset_names_match_zeds_convention() {
        assert_eq!(
            expected_asset_name(Platform::Windows, Arch::X86_64),
            "StealCode-x86_64.exe"
        );
        assert_eq!(
            expected_asset_name(Platform::MacOs, Arch::Aarch64),
            "StealCode-aarch64.dmg"
        );
        assert_eq!(
            expected_asset_name(Platform::Linux, Arch::X86_64),
            "stealcode-linux-x86_64.tar.gz"
        );
    }

    #[test]
    fn platform_arch_detection_covers_all_ci_targets() {
        assert_eq!(
            platform_arch_for("windows", "aarch64").unwrap(),
            (Platform::Windows, Arch::Aarch64)
        );
        assert_eq!(
            platform_arch_for("linux", "x86_64").unwrap(),
            (Platform::Linux, Arch::X86_64)
        );
        assert!(platform_arch_for("freebsd", "x86_64").is_err());
    }

    #[test]
    fn finds_the_matching_asset_by_exact_name() {
        let body = sample_release_json(
            "v1.0.0",
            false,
            &[("StealCode-x86_64.exe", "https://x")],
        );
        let release = parse_release_response(body.as_bytes()).unwrap();
        assert!(find_asset_by_name(&release, "StealCode-x86_64.exe").is_ok());
        assert!(
            find_asset_by_name(&release, "stealcode-linux-x86_64.tar.gz")
                .is_err()
        );
    }

    #[test]
    fn stable_channel_ignores_prereleases_nightly_accepts_them() {
        let current = Version::new(0, 3, 0);
        let body = sample_release_json("v0.4.0", true, &[]);
        let release = parse_release_response(body.as_bytes()).unwrap();

        assert_eq!(
            newer_version_available(&release, &current, ReleaseChannel::Stable)
                .unwrap(),
            None
        );
        assert_eq!(
            newer_version_available(
                &release,
                &current,
                ReleaseChannel::Nightly
            )
            .unwrap(),
            Some(Version::new(0, 4, 0))
        );
    }

    #[test]
    fn release_by_tag_url_normalizes_the_v_prefix() {
        let source = GithubReleaseSource::new("she-workss", "stealcode", None);
        assert_eq!(
            source.release_by_tag_url("v1.2.3"),
            "https://api.github.com/repos/she-workss/stealcode/releases/tags/v1.2.3"
        );
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn blocking_wrappers_work_inside_a_runtime_too() {
        // Regression test: the CLI drives the TUI under `#[tokio::main]`,
        // so the `_blocking` wrappers must not create a runtime or
        // `block_on` from the runtime's own thread (that panics with
        // "Cannot start a runtime from within a runtime"). They run on a
        // dedicated thread with their own runtime instead.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            assert!(cleanup_windows_blocking().is_ok());
            finalize_auto_update_on_quit_blocking();
        });
    }

    #[test]
    fn atomic_swap_survives_the_target_being_currently_executing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("stealcode");
        let new_binary = dir.path().join("stealcode.new");
        std::fs::write(&target, b"old").unwrap();
        std::fs::write(&new_binary, b"new").unwrap();
        let held = std::fs::File::open(&target).unwrap();
        atomic_swap(&new_binary, &target).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut { held }, &mut buf).unwrap();
        assert_eq!(buf, b"old");
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
    }
}
