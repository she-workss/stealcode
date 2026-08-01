use std::{
    path::{Path, PathBuf},
    sync::{LazyLock, OnceLock},
};

pub use utils::paths::home_dir;
use utils::{paths::SanitizedPath, rel_path::RelPath};

/// The application name, used to derive platform-specific data, config, cache,
/// and state directory paths.
pub const APP_NAME: &str = "StealCode";

/// A custom data directory override, set only by `set_custom_data_dir`.
/// This is used to override the default data directory location.
/// The directory will be created if it doesn't exist when set.
static CUSTOM_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Lowercased form of [`APP_NAME`], for use in XDG-style paths on
/// Linux/FreeBSD and the macOS `~/.config` fallback.
pub const APP_NAME_LOWERCASE: &str = {
    const LEN: usize = APP_NAME.len();
    const BYTES: [u8; LEN] = {
        let src = APP_NAME.as_bytes();
        let mut bytes = [0u8; LEN];
        let mut i = 0;
        while i < LEN {
            let b = src[i];
            assert!(b.is_ascii(), "APP_NAME must be ASCII");
            assert!(
                !b.is_ascii_control(),
                "APP_NAME must not contain control characters"
            );
            assert!(
                b != b'/' && b != b'\\',
                "APP_NAME must not contain path separators"
            );
            bytes[i] = b.to_ascii_lowercase();
            i += 1;
        }
        bytes
    };
    assert!(!APP_NAME.is_empty(), "APP_NAME must not be empty");
    match std::str::from_utf8(&BYTES) {
        Ok(s) => s,
        Err(_) => unreachable!(),
    }
};

/// The resolved data directory, combining custom override or platform defaults.
/// This is set once and cached for subsequent calls.
/// On macOS, this is `~/Library/Application Support/StealCode`.
/// On Linux/FreeBSD, this is `$XDG_DATA_HOME/stealcode`.
/// On Windows, this is `%LOCALAPPDATA%\StealCode`.
static CURRENT_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The resolved config directory, combining custom override or platform
/// defaults. This is set once and cached for subsequent calls.
/// On macOS, this is `~/.config/stealcode`.
/// On Linux/FreeBSD, this is `$XDG_CONFIG_HOME/stealcode`.
/// On Windows, this is `%APPDATA%\StealCode`.
static CONFIG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Sets a custom directory for all user data, overriding the default data
/// directory. This function must be called before any other path operations
/// that depend on the data directory, and at most once. The directory's path
/// will be canonicalized to an absolute path by a blocking FS operation. The
/// directory will be created if it doesn't exist.
///
/// # Arguments
///
/// * `dir` - The path to use as the custom data directory. This will be used as
///   the base directory for all user data, including databases, extensions, and
///   logs.
///
/// # Returns
///
/// A reference to the static `PathBuf` containing the custom data directory
/// path.
///
/// # Panics
///
/// Panics if:
/// * Called more than once
/// * Called after the data directory has been initialized (e.g., via `data_dir`
///   or `config_dir`)
/// * The directory's path cannot be canonicalized to an absolute path
/// * The directory cannot be created
pub fn set_custom_data_dir(dir: impl AsRef<Path>) -> &'static PathBuf {
    assert!(
        CUSTOM_DATA_DIR.get().is_none()
            && CURRENT_DATA_DIR.get().is_none()
            && CONFIG_DIR.get().is_none(),
        "set_custom_data_dir must be called at most once, and before data_dir \
         or config_dir are first used"
    );
    CUSTOM_DATA_DIR.get_or_init(|| {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .expect("failed to create custom data directory");
        let canonicalized = dir
            .canonicalize()
            .expect("failed to canonicalize custom data directory path");
        // On Windows, `canonicalize` produces extended-length paths prefixed
        // with `\\?\`. Strip that prefix so downstream consumers (e.g.
        // Node.js language servers) that receive derived paths as arguments
        // don't choke on the verbatim syntax.
        SanitizedPath::new(&canonicalized).as_path().to_path_buf()
    })
}

/// Returns the path to the configuration directory used by `StealCode`.
pub fn config_dir() -> &'static PathBuf {
    CONFIG_DIR.get_or_init(|| {
        if let Some(custom_dir) = CUSTOM_DATA_DIR.get() {
            return custom_dir.join("config");
        }

        cfg_select! {
            target_os = "windows" => {
                dirs::config_dir()
                    .expect("failed to determine RoamingAppData directory")
                    .join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let base_dir = if let Ok(flatpak_xdg_config) =
                    std::env::var("FLATPAK_XDG_CONFIG_HOME")
                {
                    flatpak_xdg_config.into()
                } else {
                    dirs::config_dir()
                        .expect("failed to determine XDG_CONFIG_HOME directory")
                };
                base_dir.join(APP_NAME_LOWERCASE)
            }
            _ => {
                home_dir().join(".config").join(APP_NAME_LOWERCASE)
            }
        }
    })
}

/// Returns the path to the data directory used by `StealCode`.
pub fn data_dir() -> &'static PathBuf {
    CURRENT_DATA_DIR.get_or_init(|| {
        if let Some(custom_dir) = CUSTOM_DATA_DIR.get() {
            return custom_dir.clone();
        }

        cfg_select! {
            target_os = "macos" => {
                home_dir()
                    .join("Library/Application Support")
                    .join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let data_local_dir = if let Ok(flatpak_xdg_data) =
                    std::env::var("FLATPAK_XDG_DATA_HOME")
                {
                    flatpak_xdg_data.into()
                } else {
                    dirs::data_local_dir()
                        .expect("failed to determine XDG_DATA_HOME directory")
                };
                data_local_dir.join(APP_NAME_LOWERCASE)
            }
            target_os = "windows" => {
                dirs::data_local_dir()
                    .expect("failed to determine LocalAppData directory")
                    .join(APP_NAME)
            }
            _ => {
                config_dir().clone() // Fallback
            }
        }
    })
}

/// Returns the path to the state directory used by `StealCode`.
///
/// On macOS, this is `~/.local/state/StealCode`.
/// On Linux/FreeBSD, this is `$XDG_STATE_HOME/stealcode`.
/// On Windows (and any other platform without a native "state" directory
/// convention), this currently falls back to the same path as `data_dir()`
/// (`%LOCALAPPDATA%\StealCode`).
pub fn state_dir() -> &'static PathBuf {
    static STATE_DIR: OnceLock<PathBuf> = OnceLock::new();
    STATE_DIR.get_or_init(|| {
        cfg_select! {
            target_os = "macos" => {
                home_dir().join(".local").join("state").join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let base_dir = if let Ok(flatpak_xdg_state) =
                    std::env::var("FLATPAK_XDG_STATE_HOME")
                {
                    flatpak_xdg_state.into()
                } else {
                    dirs::state_dir()
                        .expect("failed to determine XDG_STATE_HOME directory")
                };
                base_dir.join(APP_NAME_LOWERCASE)
            }
            _ => {
                dirs::data_local_dir()
                    .expect("failed to determine LocalAppData directory")
                    .join(APP_NAME)
            }
        }
    })
}

/// Returns the path to the temp directory used by `StealCode`.
pub fn temp_dir() -> &'static PathBuf {
    static TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();
    TEMP_DIR.get_or_init(|| {
        cfg_select! {
            target_os = "macos" => {
                dirs::cache_dir()
                    .expect("failed to determine cachesDirectory directory")
                    .join(APP_NAME)
            }
            target_os = "windows" => {
                dirs::cache_dir()
                    .expect("failed to determine LocalAppData directory")
                    .join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let cache_dir = if let Ok(flatpak_xdg_cache) =
                    std::env::var("FLATPAK_XDG_CACHE_HOME")
                {
                    flatpak_xdg_cache.into()
                } else {
                    dirs::cache_dir()
                        .expect("failed to determine XDG_CACHE_HOME directory")
                };
                cache_dir.join(APP_NAME_LOWERCASE)
            }
            _ => {
                home_dir().join(".cache").join(APP_NAME_LOWERCASE)
            }
        }
    })
}

/// Returns the path to the logs directory.
pub fn logs_dir() -> &'static PathBuf {
    static LOGS_DIR: OnceLock<PathBuf> = OnceLock::new();
    LOGS_DIR.get_or_init(|| {
        cfg_select! {
            target_os = "macos" => {
                home_dir().join("Library/Logs").join(APP_NAME)
            }
            _ => {
                data_dir().join("logs")
            }
        }
    })
}

/// Returns the path to the `stealcode.log` file.
pub fn log_file() -> &'static PathBuf {
    static LOG_FILE: OnceLock<PathBuf> = OnceLock::new();
    LOG_FILE
        .get_or_init(|| logs_dir().join(format!("{APP_NAME_LOWERCASE}.log")))
}

/// Returns the path to the database directory.
pub fn database_dir() -> &'static PathBuf {
    static DATABASE_DIR: OnceLock<PathBuf> = OnceLock::new();
    DATABASE_DIR.get_or_init(|| data_dir().join("db"))
}

/// Returns the path to the crashes directory, if it exists for the current
/// platform.
#[must_use]
pub fn crashes_dir() -> Option<&'static PathBuf> {
    cfg_select! {
        target_os = "macos" => {
            static CRASHES_DIR: OnceLock<PathBuf> = OnceLock::new();
            Some(CRASHES_DIR.get_or_init(|| {
                home_dir().join("Library/Logs/DiagnosticReports")
            }))
        }
        _ => {
            None
        }
    }
}

/// Returns the path to the `settings.json` file.
pub fn settings_file() -> &'static PathBuf {
    static SETTINGS_FILE: OnceLock<PathBuf> = OnceLock::new();
    SETTINGS_FILE.get_or_init(|| config_dir().join("settings.json"))
}

/// Returns the path to the `global_settings.json` file.
pub fn global_settings_file() -> &'static PathBuf {
    static GLOBAL_SETTINGS_FILE: OnceLock<PathBuf> = OnceLock::new();
    GLOBAL_SETTINGS_FILE
        .get_or_init(|| config_dir().join("global_settings.json"))
}

/// Returns the path to the `settings_backup.json` file.
pub fn settings_backup_file() -> &'static PathBuf {
    static SETTINGS_FILE: OnceLock<PathBuf> = OnceLock::new();
    SETTINGS_FILE.get_or_init(|| config_dir().join("settings_backup.json"))
}

/// Returns the path to the `auth.json` file.
pub fn auth_file() -> &'static PathBuf {
    static AUTH_FILE: OnceLock<PathBuf> = OnceLock::new();
    AUTH_FILE.get_or_init(|| config_dir().join("auth.json"))
}

/// Returns the path to the `keybindings.json` file.
pub fn keybindings_file() -> &'static PathBuf {
    static KEYBINDINGS_FILE: OnceLock<PathBuf> = OnceLock::new();
    KEYBINDINGS_FILE.get_or_init(|| config_dir().join("keybindings.json"))
}

/// Returns the path to the `keybindings_backup.json` file.
pub fn keybindings_backup_file() -> &'static PathBuf {
    static KEYBINDINGS_FILE: OnceLock<PathBuf> = OnceLock::new();
    KEYBINDINGS_FILE
        .get_or_init(|| config_dir().join("keybindings_backup.json"))
}

/// Returns the path to the extensions directory.
///
/// This is where installed extensions are stored.
pub fn plugins_dir() -> &'static PathBuf {
    static EXTENSIONS_DIR: OnceLock<PathBuf> = OnceLock::new();
    EXTENSIONS_DIR.get_or_init(|| data_dir().join("extensions"))
}

/// Returns the path to the themes directory.
///
/// This is where themes that are not provided by extensions are stored.
pub fn themes_dir() -> &'static PathBuf {
    static THEMES_DIR: OnceLock<PathBuf> = OnceLock::new();
    THEMES_DIR.get_or_init(|| config_dir().join("themes"))
}

/// Returns the path to the languages directory.
///
/// This is where language servers are downloaded to for languages built-in to
/// `StealCode`.
pub fn languages_dir() -> &'static PathBuf {
    static LANGUAGES_DIR: OnceLock<PathBuf> = OnceLock::new();
    LANGUAGES_DIR.get_or_init(|| data_dir().join("languages"))
}

/// Returns the relative path to a `.stealcode` folder within a project.
#[must_use]
pub const fn local_settings_folder_name() -> &'static str {
    ".stealcode"
}

/// Returns the relative path to a `settings.json` file within a project.
#[must_use]
pub fn local_settings_file_relative_path() -> &'static RelPath {
    static CACHED: LazyLock<&'static RelPath> =
        LazyLock::new(|| RelPath::unix(".stealcode/settings.json").unwrap());
    *CACHED
}
