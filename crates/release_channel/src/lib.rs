//! Provides the StealCode release channel (stable vs nightly).
//!
//! Independent implementation for StealCode, inspired by Zed's
//! `crates/release_channel` (GPL-3.0-or-later) but not copied from it -
//! written fresh to stay under the workspace's MIT license, and
//! deliberately dependency-light (no `gpui`) so it can be used identically
//! from the CLI, TUI, and GUI frontends - unlike Zed, which is GPUI-only
//! end to end and can afford a `gpui`-coupled version.

use std::{env, str::FromStr, sync::LazyLock};

/// "stable" | "nightly" - the raw string baked in at build time from
/// `crates/cli/RELEASE_CHANNEL`, or overridden by `STEALCODE_RELEASE_CHANNEL`
/// in debug builds only (release builds always use the baked-in file, so a
/// nightly binary can never be spoofed into reporting as stable via an env
/// var once it's actually shipped).
pub static RELEASE_CHANNEL_NAME: LazyLock<String> = LazyLock::new(|| {
    if cfg!(debug_assertions) {
        env::var("STEALCODE_RELEASE_CHANNEL")
            .unwrap_or_else(|_| compile_time_release_channel_name())
    } else {
        compile_time_release_channel_name()
    }
});

/// When `release_channel` is built in isolation by a vendoring build system
/// that can't see sibling crate directories (e.g. Nix's `crane`), the
/// relative `include_str!` below would fail to find
/// `crates/cli/RELEASE_CHANNEL`. `build.rs` detects the
/// `STEALCODE_RELEASE_CHANNEL` env var and sets this cfg so such builds can
/// pass the channel in directly instead.
#[cfg(__do_not_set_stealcode_release_channel)]
fn compile_time_release_channel_name() -> String {
    env!("STEALCODE_RELEASE_CHANNEL").trim().to_string()
}

#[cfg(not(__do_not_set_stealcode_release_channel))]
fn compile_time_release_channel_name() -> String {
    include_str!("../../cli/RELEASE_CHANNEL").trim().to_string()
}

/// The globally resolved release channel, parsed once from
/// `RELEASE_CHANNEL_NAME`. Panics at first access if the file/env var
/// contains anything other than `stable` or `nightly` - this is meant to
/// fail loudly at startup, not silently default.
pub static RELEASE_CHANNEL: LazyLock<ReleaseChannel> = LazyLock::new(|| {
    ReleaseChannel::from_str(&RELEASE_CHANNEL_NAME).unwrap_or_else(|_| {
        panic!("invalid release channel {:?}", *RELEASE_CHANNEL_NAME)
    })
});

/// A StealCode release channel. Stable and nightly are separate,
/// side-by-side-installable builds (different `AppId`/AUMID in
/// `stealcode.iss`), not a runtime toggle - you pick one when you download
/// the installer, and self-update only ever moves you within that channel.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ReleaseChannel {
    #[default]
    Stable,
    Nightly,
}

impl ReleaseChannel {
    pub const ALL: [ReleaseChannel; 2] =
        [ReleaseChannel::Stable, ReleaseChannel::Nightly];

    /// Returns the channel this binary was built for.
    #[must_use]
    pub fn current() -> Self {
        *RELEASE_CHANNEL
    }

    /// Nightly accepts GitHub prereleases as valid updates; stable never
    /// does.
    #[must_use]
    pub const fn accepts_prereleases(self) -> bool {
        matches!(self, Self::Nightly)
    }

    #[must_use]
    pub const fn poll_interval(self) -> std::time::Duration {
        match self {
            Self::Stable => std::time::Duration::from_secs(60 * 60),
            Self::Nightly => std::time::Duration::from_secs(15 * 60),
        }
    }

    /// Human-readable name shown in the UI / window title.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Stable => "StealCode",
            Self::Nightly => "StealCode Nightly",
        }
    }

    /// Programmatic name, used in file names, registry keys, log lines.
    #[must_use]
    pub const fn dev_name(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Nightly => "nightly",
        }
    }

    /// The Windows AppUserModelID for this channel - must be distinct per
    /// channel so stable and nightly installs get separate taskbar/toast
    /// identities and don't fight over the same jump list.
    #[must_use]
    pub const fn app_user_model_id(self) -> &'static str {
        match self {
            Self::Stable => "he-thinks.StealCode",
            Self::Nightly => "he-thinks.StealCode.Nightly",
        }
    }
}

/// Error returned by [`ReleaseChannel::from_str`] for any string other than
/// `"stable"` or `"nightly"`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InvalidReleaseChannel;

impl std::fmt::Display for InvalidReleaseChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "invalid release channel: expected \"stable\" or \"nightly\"",
        )
    }
}

impl std::error::Error for InvalidReleaseChannel {}

impl FromStr for ReleaseChannel {
    type Err = InvalidReleaseChannel;

    fn from_str(channel: &str) -> Result<Self, Self::Err> {
        Ok(match channel {
            "stable" => Self::Stable,
            "nightly" => Self::Nightly,
            _ => return Err(InvalidReleaseChannel),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_channel_names() {
        assert_eq!(
            ReleaseChannel::from_str("stable").unwrap(),
            ReleaseChannel::Stable
        );
        assert_eq!(
            ReleaseChannel::from_str("nightly").unwrap(),
            ReleaseChannel::Nightly
        );
    }

    #[test]
    fn rejects_unknown_channel_names() {
        assert!(ReleaseChannel::from_str("preview").is_err());
        assert!(ReleaseChannel::from_str("").is_err());
        assert!(ReleaseChannel::from_str("Stable").is_err());
    }

    #[test]
    fn nightly_accepts_prereleases_stable_does_not() {
        assert!(!ReleaseChannel::Stable.accepts_prereleases());
        assert!(ReleaseChannel::Nightly.accepts_prereleases());
    }

    #[test]
    fn nightly_polls_more_often_than_stable() {
        assert!(
            ReleaseChannel::Nightly.poll_interval()
                < ReleaseChannel::Stable.poll_interval()
        );
    }

    #[test]
    fn app_user_model_ids_differ_per_channel() {
        assert_ne!(
            ReleaseChannel::Stable.app_user_model_id(),
            ReleaseChannel::Nightly.app_user_model_id()
        );
    }

    #[test]
    fn default_channel_is_stable() {
        assert_eq!(ReleaseChannel::default(), ReleaseChannel::Stable);
    }
}
