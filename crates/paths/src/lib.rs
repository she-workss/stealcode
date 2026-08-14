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

/// The resolved data directory, cached after first use.
/// macOS `~/Library/Application Support/StealCode`; Linux/FreeBSD
/// `$XDG_DATA_HOME/stealcode`; Windows `%LOCALAPPDATA%\StealCode`.
static CURRENT_DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The resolved config directory, cached after first use.
/// macOS `~/.config/stealcode`; Linux/FreeBSD `$XDG_CONFIG_HOME/stealcode`;
/// Windows `%APPDATA%\StealCode`.
static CONFIG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Sets a custom base directory for all user data, overriding platform
/// defaults. Must be called at most once, before `data_dir`/`config_dir`
/// are first used; the directory is created if missing.
///
/// # Panics
/// Panics if called more than once, after the data/config dirs are
/// initialized, or if the directory cannot be created or canonicalized.
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

/// A `dirs`-crate lookup with a fallback relative to the home directory,
/// so a missing platform dir can never panic the `OnceLock` caches below.
fn dir_or_home(dir: Option<PathBuf>, relative: &str) -> PathBuf {
    dir.unwrap_or_else(|| home_dir().join(relative))
}

/// Returns the path to the configuration directory used by `StealCode`.
pub fn config_dir() -> &'static PathBuf {
    CONFIG_DIR.get_or_init(|| {
        if let Some(custom_dir) = CUSTOM_DATA_DIR.get() {
            return custom_dir.join("config");
        }

        cfg_select! {
            target_os = "windows" => {
                dir_or_home(dirs::config_dir(), "AppData/Roaming")
                    .join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let base_dir = if let Ok(flatpak_xdg_config) =
                    std::env::var("FLATPAK_XDG_CONFIG_HOME")
                {
                    flatpak_xdg_config.into()
                } else {
                    dir_or_home(dirs::config_dir(), ".config")
                };
                base_dir.join(APP_NAME_LOWERCASE)
            }
            _ => home_dir().join(".config").join(APP_NAME_LOWERCASE),
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
            target_os = "macos" => home_dir()
                .join("Library/Application Support")
                .join(APP_NAME),
            any(target_os = "linux", target_os = "freebsd") => {
                let data_local_dir = if let Ok(flatpak_xdg_data) =
                    std::env::var("FLATPAK_XDG_DATA_HOME")
                {
                    flatpak_xdg_data.into()
                } else {
                    dir_or_home(dirs::data_local_dir(), ".local/share")
                };
                data_local_dir.join(APP_NAME_LOWERCASE)
            }
            target_os = "windows" => {
                dir_or_home(dirs::data_local_dir(), "AppData/Local")
                    .join(APP_NAME)
            }
            _ => config_dir().clone(),
        }
    })
}

/// Returns the path to the state directory used by `StealCode`.
/// macOS `~/.local/state/StealCode`; Linux/FreeBSD `$XDG_STATE_HOME/stealcode`;
/// falls back to `data_dir()` on other platforms.
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
                    dir_or_home(dirs::state_dir(), ".local/state")
                };
                base_dir.join(APP_NAME_LOWERCASE)
            }
            _ => dir_or_home(dirs::data_local_dir(), "AppData/Local")
                .join(APP_NAME),
        }
    })
}

/// Returns the path to the temp directory used by `StealCode`.
pub fn temp_dir() -> &'static PathBuf {
    static TEMP_DIR: OnceLock<PathBuf> = OnceLock::new();
    TEMP_DIR.get_or_init(|| {
        cfg_select! {
            target_os = "macos" => {
                dir_or_home(dirs::cache_dir(), "Library/Caches").join(APP_NAME)
            }
            target_os = "windows" => {
                dir_or_home(dirs::cache_dir(), "AppData/Local").join(APP_NAME)
            }
            any(target_os = "linux", target_os = "freebsd") => {
                let cache_dir = if let Ok(flatpak_xdg_cache) =
                    std::env::var("FLATPAK_XDG_CACHE_HOME")
                {
                    flatpak_xdg_cache.into()
                } else {
                    dir_or_home(dirs::cache_dir(), ".cache")
                };
                cache_dir.join(APP_NAME_LOWERCASE)
            }
            _ => home_dir().join(".cache").join(APP_NAME_LOWERCASE),
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
pub const fn crashes_dir() -> Option<&'static PathBuf> {
    cfg_select! {
        target_os = "macos" => {
            static CRASHES_DIR: OnceLock<PathBuf> = OnceLock::new();
            Some(CRASHES_DIR.get_or_init(|| {
                home_dir().join("Library/Logs/DiagnosticReports")
            }))
        }
        _ => None,
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
pub fn plugins_dir() -> &'static PathBuf {
    static EXTENSIONS_DIR: OnceLock<PathBuf> = OnceLock::new();
    EXTENSIONS_DIR.get_or_init(|| data_dir().join("extensions"))
}

/// Returns the path to the themes directory (themes not provided by
/// extensions).
pub fn themes_dir() -> &'static PathBuf {
    static THEMES_DIR: OnceLock<PathBuf> = OnceLock::new();
    THEMES_DIR.get_or_init(|| config_dir().join("themes"))
}

/// Returns the path to the languages directory, where language servers for
/// built-in languages are downloaded.
pub fn languages_dir() -> &'static PathBuf {
    static LANGUAGES_DIR: OnceLock<PathBuf> = OnceLock::new();
    LANGUAGES_DIR.get_or_init(|| data_dir().join("languages"))
}

/// Hugging Face Hub repository holding the speech model.
pub const MODEL_REPO: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";
/// GGUF checkpoint file inside the model repository.
pub const GGUF_FILE: &str = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf";

/// Returns the path to the speech model directory.
pub fn model_dir() -> &'static PathBuf {
    static MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
    MODEL_DIR.get_or_init(|| {
        data_dir().join("models").join("nvidia").join("nemotron")
    })
}

/// Returns the full path of the speech model GGUF checkpoint.
pub fn model_path() -> &'static PathBuf {
    static MODEL_PATH: OnceLock<PathBuf> = OnceLock::new();
    MODEL_PATH.get_or_init(|| model_dir().join(GGUF_FILE))
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
