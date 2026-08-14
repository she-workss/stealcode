use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
pub use config::*;
use paths::{
    auth_file, global_settings_file, local_settings_folder_name, settings_file,
};
use rustc_hash::FxHashMap;
use serde_json::Value;
use tracing::warn;

pub mod config;
pub mod keybindings;
pub mod keystroke;
pub mod log_level;
pub mod permission;
pub mod plugin;
pub mod state;
pub mod tui;

/// Reads and parses a JSON file, returning `None` (with a warning) if the
/// file is missing or malformed so callers can fall back to defaults.
fn read_json_file(path: &Path) -> Option<Value> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!("failed to read {}: {e}", path.display());
            return None;
        }
    };
    match serde_json::from_str(&raw) {
        Ok(value) => Some(value),
        Err(e) => {
            warn!("failed to parse {} as JSON: {e}", path.display());
            None
        }
    }
}

/// Recursively merges `override_` into `base`: objects merge key by key,
/// any other value in `override_` replaces the base value wholesale.
fn merge_json(base_val: &mut Value, override_val: Value) {
    match (base_val, override_val) {
        (Value::Object(base_map), Value::Object(over_map)) => {
            for (k, v) in over_map {
                if let Some(base_val) = base_map.get_mut(&k) {
                    merge_json(base_val, v);
                } else {
                    base_map.insert(k, v);
                }
            }
        }
        (base, other) => *base = other,
    }
}

/// Reads each of `paths` that exists and deep-merges them in order, so later
/// paths take priority over earlier ones (missing/unparseable ones are
/// skipped).
fn merge_json_layers(paths: &[&Path]) -> Value {
    let mut value = Value::Object(serde_json::Map::new());
    for path in paths {
        if let Some(layer) = read_json_file(path) {
            merge_json(&mut value, layer);
        }
    }
    value
}

/// Writes `data` to `path` atomically via a `.tmp` file in
/// [`std::env::temp_dir`] plus rename, so the TUI filesystem watcher never
/// sees spurious `.tmp` events inside the config directory.
fn write_atomic(path: &Path, data: &str) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config");
    let tmp_dir = std::env::temp_dir().join("stealcode");
    fs::create_dir_all(&tmp_dir).with_context(|| {
        format!("failed to create tmp dir {}", tmp_dir.display())
    })?;
    let tmp = tmp_dir.join(format!("{file_name}.tmp"));
    fs::write(&tmp, data)
        .with_context(|| format!("failed to write to {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!("failed to rename {} -> {}", tmp.display(), path.display())
    })
}

/// Ensures `path`'s parent directory exists, then returns `path` unchanged.
fn ensure_parent_dir(path: &'static Path) -> Result<&'static Path> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create directory {}", parent.display())
        })?;
    }
    Ok(path)
}

/// Picks the write target for the two global settings tiers: the user
/// settings file if it exists, else the defaults file (creating its parent).
/// Writes never target the project-local `.stealcode/settings.json`.
fn resolve_write_target(
    user_settings: &'static Path,
    defaults: &'static Path,
) -> Result<&'static Path> {
    if user_settings.exists() {
        return Ok(user_settings);
    }
    ensure_parent_dir(defaults)
}

/// Returns the path to `<cwd>/.stealcode/settings.json` if that file exists.
///
/// Like VS Code's `.vscode` semantics, only the exact `cwd` is checked -
/// parent directories are not walked.
#[must_use]
pub fn find_local_settings_file(cwd: &Path) -> Option<PathBuf> {
    let candidate =
        cwd.join(local_settings_folder_name()).join("settings.json");
    candidate.is_file().then_some(candidate)
}

fn default_settings_value() -> Result<Value> {
    let settings = Settings {
        providers: default_providers(),
        ..Default::default()
    };
    let mut value = serde_json::to_value(settings)?;
    if let Value::Object(ref mut m) = value {
        m.insert(
            "$schema".into(),
            Value::String("https://opencode.ai/config.json".into()),
        );
    }
    Ok(value)
}

/// Creates missing config files on first run: the global defaults file, the
/// user overrides file (written as `{}`), and the auth file (written as `{}`).
/// Call once at startup, before [`load_settings`].
pub fn init_config() -> Result<()> {
    let global = global_settings_file();
    if let Some(parent) = global.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create {}", parent.display())
        })?;
    }
    if !global.exists() {
        let data = serde_json::to_string_pretty(&default_settings_value()?)?;
        write_atomic(global, &data)?;
    }

    let user = settings_file();
    if !user.exists() {
        let data = serde_json::to_string_pretty(&Value::Object(
            serde_json::Map::new(),
        ))?;
        write_atomic(user, &data)?;
    }

    let auth = auth_file();
    if !auth.exists() {
        // `AuthSettings` is `#[serde(flatten)]`, so an empty auth file is
        // `{}`, not `{"providers": {}}`.
        let data =
            serde_json::to_string_pretty(&config::AuthSettings::default())?;
        write_atomic(auth, &data)?;
    }

    Ok(())
}

/// Default providers injected when no providers are configured.
fn default_providers() -> FxHashMap<String, ProviderEntry> {
    let mut providers = FxHashMap::default();
    providers.insert(
        "openai".into(),
        ProviderEntry {
            url: "https://api.openai.com/v1".into(),
            provider_type: "openai".into(),
        },
    );
    providers.insert(
        "opencode-go".into(),
        ProviderEntry {
            url: "https://opencode.ai/zen/go/v1".into(),
            provider_type: "opencode".into(),
        },
    );
    providers.insert(
        "opencode-zen".into(),
        ProviderEntry {
            url: "https://opencode.ai/zen/v1".into(),
            provider_type: "opencode".into(),
        },
    );
    providers.insert(
        "deepseek".into(),
        ProviderEntry {
            url: "https://api.deepseek.com/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "openrouter".into(),
        ProviderEntry {
            url: "https://openrouter.ai/api/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "moonshot".into(),
        ProviderEntry {
            url: "https://api.moonshot.cn/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "minimax".into(),
        ProviderEntry {
            url: "https://api.minimax.chat/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "zai".into(),
        ProviderEntry {
            url: "https://api.zer.ai/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "nvidia".into(),
        ProviderEntry {
            url: "https://integrate.api.nvidia.com/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "huggingface".into(),
        ProviderEntry {
            url: "https://api-inference.huggingface.co/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "ollama".into(),
        ProviderEntry {
            url: "http://localhost:11434/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "lm-studio".into(),
        ProviderEntry {
            url: "http://localhost:1234/v1".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers.insert(
        "github-copilot".into(),
        ProviderEntry {
            url: "https://api.githubcopilot.com".into(),
            provider_type: "openai-compatible".into(),
        },
    );
    providers
}

/// Loads settings by merging layers, lowest priority first: factory defaults,
/// then the user's overrides, then `<cwd>/.stealcode/settings.json` (highest;
/// only the exact `cwd` is checked, parents are not walked).
#[must_use]
pub fn load_settings(cwd: &Path) -> Settings {
    let local_settings = find_local_settings_file(cwd);

    let mut layers: Vec<&Path> =
        vec![global_settings_file().as_path(), settings_file().as_path()];
    if let Some(local_path) = &local_settings {
        layers.push(local_path.as_path());
    }

    let value = merge_json_layers(&layers);
    let mut settings: Settings =
        serde_json::from_value(value).unwrap_or_else(|e| {
            warn!("failed to deserialize settings, using defaults: {e}");
            Settings::default()
        });
    if settings.providers.is_empty() {
        settings.providers = default_providers();
    }
    settings
}

#[must_use]
pub fn load_auth() -> config::AuthSettings {
    let value = read_json_file(auth_file())
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    serde_json::from_value(value).unwrap_or_else(|e| {
        warn!("failed to deserialize auth, using defaults: {e}");
        config::AuthSettings::default()
    })
}

pub fn patch_settings_value(f: impl FnOnce(&mut Value)) -> Result<()> {
    let target = resolve_write_target(
        settings_file().as_path(),
        global_settings_file().as_path(),
    )?;
    let mut value = read_json_file(target)
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    f(&mut value);
    write_atomic(target, &serde_json::to_string_pretty(&value)?)
}

pub fn save_settings(settings: &Settings) -> Result<()> {
    let target = resolve_write_target(
        settings_file().as_path(),
        global_settings_file().as_path(),
    )?;
    write_atomic(target, &serde_json::to_string_pretty(settings)?)
}

pub fn save_auth(auth: &config::AuthSettings) -> Result<()> {
    let target = ensure_parent_dir(auth_file().as_path())?;
    write_atomic(target, &serde_json::to_string_pretty(auth)?)
}

pub fn save_models(models: &[Value], models_file: &str) -> Result<()> {
    let path = Path::new(models_file);
    write_atomic(path, &serde_json::to_string_pretty(models)?)
}

#[cfg(test)]
mod tests {
    use paths::local_settings_folder_name;

    use super::*;

    fn unique_temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "stealcode-settings-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is before unix epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn merge_json_recursively_merges_objects() {
        let mut base = serde_json::json!({
            "a": 1,
            "nested": { "x": 1, "y": 2 },
            "untouched": "base"
        });
        let override_ = serde_json::json!({
            "a": 2,
            "nested": { "y": 20, "z": 30 }
        });
        merge_json(&mut base, override_);
        assert_eq!(
            base,
            serde_json::json!({
                "a": 2,
                "nested": { "x": 1, "y": 20, "z": 30 },
                "untouched": "base"
            })
        );
    }

    #[test]
    fn merge_json_override_replaces_non_object_values_wholesale() {
        let mut base = serde_json::json!({ "list": [1, 2, 3], "value": "old" });
        let override_ = serde_json::json!({ "list": [9], "value": null });
        merge_json(&mut base, override_);
        assert_eq!(base, serde_json::json!({ "list": [9], "value": null }));
    }

    #[test]
    fn merge_json_layers_applies_priority_low_to_high() {
        let root = unique_temp_dir("merge-layers");
        fs::create_dir_all(&root).expect("failed to create test dir");

        let low = root.join("low.json");
        let high = root.join("high.json");
        let missing = root.join("missing.json");

        fs::write(&low, r#"{"a": 1, "b": 1}"#).expect("write low");
        fs::write(&high, r#"{"b": 2, "c": 3}"#).expect("write high");

        let value = merge_json_layers(&[
            low.as_path(),
            missing.as_path(),
            high.as_path(),
        ]);

        assert_eq!(value, serde_json::json!({ "a": 1, "b": 2, "c": 3 }));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_json_file_returns_none_for_missing_file() {
        let path =
            unique_temp_dir("read-json-missing").join("does-not-exist.json");
        assert_eq!(read_json_file(&path), None);
    }

    #[test]
    fn read_json_file_returns_none_for_invalid_json() {
        let root = unique_temp_dir("read-json-invalid");
        fs::create_dir_all(&root).expect("failed to create test dir");
        let path = root.join("invalid.json");
        fs::write(&path, "{ not valid json").expect("write invalid json");

        assert_eq!(read_json_file(&path), None);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_json_file_parses_valid_json() {
        let root = unique_temp_dir("read-json-valid");
        fs::create_dir_all(&root).expect("failed to create test dir");
        let path = root.join("valid.json");
        fs::write(&path, r#"{"a": 1}"#).expect("write valid json");

        assert_eq!(read_json_file(&path), Some(serde_json::json!({ "a": 1 })));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn default_settings_value_round_trips_and_has_schema_and_providers() {
        let value = default_settings_value().unwrap();
        assert_eq!(
            value.get("$schema").and_then(Value::as_str),
            Some("https://opencode.ai/config.json")
        );

        let settings: Settings = serde_json::from_value(value).expect(
            "default settings value should deserialize back into Settings",
        );
        assert!(!settings.providers.is_empty());
    }

    #[test]
    fn find_local_settings_file_finds_file_in_cwd() {
        let root = unique_temp_dir("find-local-cwd");
        let local_dir = root.join(local_settings_folder_name());
        fs::create_dir_all(&local_dir)
            .expect("failed to create .stealcode dir");
        fs::write(local_dir.join("settings.json"), "{}")
            .expect("failed to write settings.json");

        let found = find_local_settings_file(&root);
        assert_eq!(found, Some(local_dir.join("settings.json")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn find_local_settings_file_does_not_walk_up_to_parent() {
        let root = unique_temp_dir("find-local-no-walk");
        let child = root.join("child");
        fs::create_dir_all(&child).expect("failed to create child dir");

        // Place the settings file in the parent only, not in child.
        let local_dir = root.join(local_settings_folder_name());
        fs::create_dir_all(&local_dir)
            .expect("failed to create .stealcode dir");
        fs::write(local_dir.join("settings.json"), "{}")
            .expect("failed to write settings.json");

        // Starting from child - must NOT find the parent's settings.
        let found = find_local_settings_file(&child);
        assert_eq!(found, None);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn find_local_settings_file_returns_none_when_absent() {
        let root = unique_temp_dir("find-local-absent");
        fs::create_dir_all(&root).expect("failed to create test dir");

        assert_eq!(find_local_settings_file(&root), None);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn find_local_settings_file_returns_none_for_directory_not_file() {
        // Edge case: `.stealcode/settings.json` exists but as a directory.
        let root = unique_temp_dir("find-local-dir-not-file");
        let fake_file = root
            .join(local_settings_folder_name())
            .join("settings.json");
        fs::create_dir_all(&fake_file)
            .expect("failed to create settings.json as dir");

        assert_eq!(find_local_settings_file(&root), None);

        let _ = fs::remove_dir_all(&root);
    }
}
