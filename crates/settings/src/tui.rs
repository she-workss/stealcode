use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ScrollAcceleration {
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffStyle {
    Auto,
    Stacked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScrollSpeed {
    Number(f64),
    Special(ScrollSpeedSpecial),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScrollSpeedSpecial {
    #[serde(rename = "NaN")]
    Nan,
    #[serde(rename = "-Infinity")]
    NegInfinity,
    #[serde(rename = "+Infinity")]
    PosInfinity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiConfig {
    #[serde(rename = "$schema")]
    pub schema: Option<String>,
    pub theme: String,
    pub leader_timeout: u64,
    pub plugin: Vec<String>,
    pub plugin_enabled: Option<FxHashMap<String, bool>>,
    pub scroll_speed: Option<ScrollSpeed>,
    pub scroll_acceleration: Option<ScrollAcceleration>,
    pub diff_style: Option<DiffStyle>,
    pub mouse: Option<bool>,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            schema: None,
            theme: String::from("StealCode"),
            leader_timeout: 1000,
            plugin: Vec::new(),
            plugin_enabled: None,
            scroll_speed: None,
            scroll_acceleration: None,
            diff_style: None,
            mouse: Some(true),
        }
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn tui_config_full_roundtrip() {
        let json = json!({
            "theme": "catppuccin-mocha",
            "mouse": false,
            "diff_style": "stacked",
            "scroll_speed": 2.5,
            "scroll_acceleration": { "enabled": true },
            "keybinds": {
                "app_exit": "ctrl+q",
                "input_submit": "enter"
            },
            "plugin_enabled": {
                "stealcode-vim": true,
                "stealcode-legacy": false
            }
        });

        let config: TuiConfig = serde_json::from_value(json).unwrap();

        assert_eq!(config.theme.as_str(), "catppuccin-mocha");
        assert_eq!(config.mouse, Some(false));
        assert_eq!(config.diff_style, Some(DiffStyle::Stacked));
        assert!(
            matches!(config.scroll_speed, Some(ScrollSpeed::Number(n)) if (n - 2.5).abs() < 0.1)
        );
        assert!(config.scroll_acceleration.unwrap().enabled);

        let pe = config.plugin_enabled.unwrap();
        assert!(pe["stealcode-vim"]);
        assert!(!pe["stealcode-legacy"]);
    }

    #[test]
    fn scroll_speed_special_values() {
        for val in ["NaN", "-Infinity", "+Infinity"] {
            let json = json!({ "scroll_speed": val });
            let config: TuiConfig = serde_json::from_value(json).unwrap();
            assert!(matches!(
                config.scroll_speed,
                Some(ScrollSpeed::Special(_))
            ));
        }
    }
}
