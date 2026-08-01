use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::skip_serializing_none;

use crate::{permission::PermissionSettings, plugin::PluginEntry};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShareMode {
    Manual,
    Auto,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AutoUpdate {
    Enabled(bool),
    Notify(AutoUpdateNotify),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoUpdateNotify {
    #[serde(rename = "notify")]
    Notify,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentMode {
    Subagent,
    Primary,
    All,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentSettings {
    pub model: Option<String>,
    pub variant: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub prompt: Option<String>,
    pub disable: Option<bool>,
    pub description: Option<String>,
    pub mode: Option<AgentMode>,
    pub hidden: Option<bool>,
    pub options: Option<FxHashMap<String, Value>>,
    pub steps: Option<u64>,
    pub permission: Option<PermissionSettings>,
    #[serde(flatten)]
    pub extra: FxHashMap<String, Value>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentMap {
    pub build: Option<AgentSettings>,
    pub plan: Option<AgentSettings>,
    pub general: Option<AgentSettings>,
    pub explore: Option<AgentSettings>,
    pub title: Option<AgentSettings>,
    pub summary: Option<AgentSettings>,
    pub compaction: Option<AgentSettings>,
    pub scout: Option<AgentSettings>,
    #[serde(flatten)]
    pub extra: FxHashMap<String, AgentSettings>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerSettings {
    pub port: Option<u16>,
    pub hostname: Option<String>,
    pub mdns: Option<bool>,
    pub mdns_domain: Option<String>,
    pub cors: Option<Vec<String>>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillsSettings {
    pub paths: Option<Vec<String>>,
    pub urls: Option<Vec<String>>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WatcherSettings {
    pub ignore: Option<Vec<String>>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSettings {
    pub template: String,
    pub description: Option<String>,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub subtask: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopApplicationSettings {
    pub theme: String,
    pub scrollbar: String,
    pub font_size: u8,
    pub border_radius: u8,
}

impl Default for DesktopApplicationSettings {
    fn default() -> Self {
        Self {
            theme: String::from("StealCode"),
            scrollbar: String::from("always"),
            font_size: 14,
            border_radius: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowSettings {
    pub width: f32,
    pub height: f32,
    pub min_size_width: f32,
    pub min_size_height: f32,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            min_size_width: 600.0,
            min_size_height: 400.0,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopSettings {
    pub application: DesktopApplicationSettings,
    pub window: WindowSettings,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetrySettings {
    pub level: String,
    pub file_path: String,
}

impl Default for TelemetrySettings {
    fn default() -> Self {
        Self {
            level: String::from("info"),
            file_path: String::from("stealcode.log"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProviderEntry {
    pub url: String,
    #[serde(default)]
    pub provider_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderModelEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAuth {
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthSettings {
    #[serde(flatten)]
    pub providers: FxHashMap<String, ProviderAuth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    #[serde(rename = "$schema")]
    pub schema: Option<String>,
    pub shell: Option<String>,
    pub server: Option<ServerSettings>,
    pub command: Option<FxHashMap<String, CommandSettings>>,
    pub skills: Option<SkillsSettings>,
    pub watcher: Option<WatcherSettings>,
    pub snapshot: Option<bool>,
    #[serde(default)]
    pub plugin: Vec<PluginEntry>,
    pub share: Option<ShareMode>,
    pub autoupdate: Option<AutoUpdate>,
    pub disabled_providers: Option<Vec<String>>,
    pub enabled_providers: Option<Vec<String>>,
    pub model: Option<String>,
    pub small_model: Option<String>,
    pub default_agent: Option<String>,
    pub username: Option<String>,
    pub agent: Option<AgentMap>,
    #[serde(default)]
    pub desktop: DesktopSettings,
    #[serde(default)]
    pub telemetry: TelemetrySettings,
    #[serde(default)]
    pub tui: crate::tui::TuiConfig,
    #[serde(default)]
    pub color_customizations: FxHashMap<String, FxHashMap<String, String>>,
    #[serde(default)]
    pub providers: FxHashMap<String, ProviderEntry>,
    /// Whether StealCode checks for updates automatically in the
    /// background. Manual checks ("Check for updates now") always work
    /// regardless of this setting. Default: `true`.
    pub auto_update: Option<bool>,
}
