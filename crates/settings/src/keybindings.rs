use paths::keybindings_file;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::keystroke::Keystroke;

/// A single keybinding entry, matching the VS Code keybindings.json schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeybindingEntry {
    /// Human-readable key description, e.g. `ctrl+enter`.
    pub key: String,
    /// Command id to execute when the key is pressed.
    pub command: String,
    /// Optional context expression that must evaluate to true for the binding
    /// to be active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
}

/// The two keybinding domains.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Keybindings {
    #[serde(default)]
    pub tui: Vec<KeybindingEntry>,
    #[serde(default)]
    pub desktop: Vec<KeybindingEntry>,
}

impl Keybindings {
    /// Returns the first [`KeybindingEntry`] in `self.tui` whose `key` parses
    /// to the given `keystroke` and whose `when` context is satisfied by
    /// `context`.
    #[must_use]
    pub fn match_tui(
        &self,
        keystroke: &Keystroke,
        context: &KeyContext,
    ) -> Option<&KeybindingEntry> {
        Self::match_in(&self.tui, keystroke, context)
    }

    /// Returns the first `key` string in `self.tui` that maps to `command`
    /// (ignoring `when`).  Used for generating UI hints.
    #[must_use]
    pub fn find_tui_key(&self, command: &str) -> Option<&str> {
        self.tui
            .iter()
            .find(|e| e.command == command)
            .map(|e| e.key.as_str())
    }

    fn match_in<'a>(
        entries: &'a [KeybindingEntry],
        keystroke: &Keystroke,
        context: &KeyContext,
    ) -> Option<&'a KeybindingEntry> {
        entries.iter().find(|entry| {
            let Ok(parsed) = Keystroke::parse(&entry.key) else {
                return false;
            };
            parsed == *keystroke
                && entry.when.as_deref().is_none_or(|w| eval_when(w, context))
        })
    }
}

/// Runtime context tags used to evaluate `when` expressions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyContext {
    pub chat: bool,
    pub loading: bool,
    pub palette_open: bool,
    pub streaming: bool,
}

/// Evaluates a `when` expression: `&&`-separated named terms against
/// `ctx`, each optionally negated with a leading `!` (e.g. `"chat &&
/// !streaming"`).
fn eval_when(expr: &str, ctx: &KeyContext) -> bool {
    let terms: Vec<&str> = expr.split("&&").map(str::trim).collect();
    for term in terms {
        if term.is_empty() {
            continue;
        }
        let negated = term.starts_with('!');
        let name = if negated { &term[1..] } else { term };
        let value = match name {
            "chat" => ctx.chat,
            "loading" => ctx.loading,
            "paletteOpen" => ctx.palette_open,
            "streaming" => ctx.streaming,
            _ => false,
        };
        if negated == value {
            return false;
        }
    }
    true
}

/// Loads keybindings from disk.  If the file does not exist it is created with
/// the default bindings and those are returned.
#[must_use]
pub fn load_keybindings() -> Keybindings {
    let keybindings_file = keybindings_file();
    if keybindings_file.exists() {
        match std::fs::read_to_string(keybindings_file) {
            Ok(text) => match serde_json::from_str::<Keybindings>(&text) {
                Ok(kb) => {
                    info!(
                        "loaded keybindings from {}",
                        keybindings_file.display()
                    );
                    return kb;
                }
                Err(e) => {
                    warn!(
                        "failed to parse keybindings from {}: {e}, using defaults",
                        keybindings_file.display()
                    );
                }
            },
            Err(e) => {
                warn!(
                    "failed to read keybindings from {}: {e}, using defaults",
                    keybindings_file.display()
                );
            }
        }
    }

    let defaults = default_keybindings();
    if let Err(e) = save_keybindings(&defaults) {
        warn!("failed to write default keybindings: {e}");
    }
    defaults
}

/// Persists keybindings to `~/.stealcode/keybindings.json`.
pub fn save_keybindings(kb: &Keybindings) -> std::io::Result<()> {
    let keybindings_file = keybindings_file();
    if let Some(parent) = keybindings_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(kb).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    })?;
    std::fs::write(keybindings_file, text)?;
    info!("saved keybindings to {}", keybindings_file.display());
    Ok(())
}

/// Default keybindings that mirror the currently hard-coded shortcuts.
#[must_use]
pub fn default_keybindings() -> Keybindings {
    Keybindings {
        tui: vec![
            // Global
            KeybindingEntry {
                key: String::from("ctrl+c"),
                command: String::from("stealcode.quit"),
                when: None,
            },
            KeybindingEntry {
                key: String::from("ctrl+p"),
                command: String::from("stealcode.togglePalette"),
                when: None,
            },
            // Palette
            KeybindingEntry {
                key: String::from("escape"),
                command: String::from("stealcode.paletteBack"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("up"),
                command: String::from("stealcode.paletteUp"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("down"),
                command: String::from("stealcode.paletteDown"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("enter"),
                command: String::from("stealcode.paletteConfirm"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("backspace"),
                command: String::from("stealcode.paletteSearchDelete"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+backspace"),
                command: String::from(
                    "stealcode.paletteSearchWordDeleteBackward",
                ),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("left"),
                command: String::from("stealcode.paletteSearchCursorLeft"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("right"),
                command: String::from("stealcode.paletteSearchCursorRight"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+left"),
                command: String::from("stealcode.paletteSearchWordLeft"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+right"),
                command: String::from("stealcode.paletteSearchWordRight"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+x m"),
                command: String::from("stealcode.palette.model"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+x p"),
                command: String::from("stealcode.palette.provider"),
                when: Some(String::from("paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+x t"),
                command: String::from("stealcode.palette.theme"),
                when: Some(String::from("paletteOpen")),
            },
            // Loading screen
            KeybindingEntry {
                key: String::from("tab"),
                command: String::from("stealcode.toggleInputMode"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+backspace"),
                command: String::from("stealcode.deleteWordBackward"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("alt+backspace"),
                command: String::from("stealcode.deleteWordBackward"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+delete"),
                command: String::from("stealcode.deleteWordForward"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("alt+delete"),
                command: String::from("stealcode.deleteWordForward"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("shift+enter"),
                command: String::from("stealcode.newLine"),
                when: Some(String::from("loading && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("enter"),
                command: String::from("stealcode.sendChat"),
                when: Some(String::from(
                    "loading && !streaming && !paletteOpen",
                )),
            },
            // Chat screen
            KeybindingEntry {
                key: String::from("tab"),
                command: String::from("stealcode.toggleInputMode"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+backspace"),
                command: String::from("stealcode.deleteWordBackward"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("alt+backspace"),
                command: String::from("stealcode.deleteWordBackward"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+delete"),
                command: String::from("stealcode.deleteWordForward"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("alt+delete"),
                command: String::from("stealcode.deleteWordForward"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("pageup"),
                command: String::from("stealcode.pageUp"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("pagedown"),
                command: String::from("stealcode.pageDown"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("up"),
                command: String::from("stealcode.inputUp"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("down"),
                command: String::from("stealcode.inputDown"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+up"),
                command: String::from("stealcode.scrollUp"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("ctrl+down"),
                command: String::from("stealcode.scrollDown"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("shift+enter"),
                command: String::from("stealcode.newLine"),
                when: Some(String::from("chat && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("enter"),
                command: String::from("stealcode.sendChat"),
                when: Some(String::from("chat && !streaming && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("enter"),
                command: String::from("stealcode.noop"),
                when: Some(String::from("chat && streaming && !paletteOpen")),
            },
            KeybindingEntry {
                key: String::from("escape"),
                command: String::from("stealcode.cancelStream"),
                when: Some(String::from("chat && streaming && !paletteOpen")),
            },
        ],
        desktop: vec![],
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_when_simple() {
        let ctx = KeyContext {
            chat: true,
            ..KeyContext::default()
        };
        assert!(eval_when("chat", &ctx));
        assert!(!eval_when("!chat", &ctx));
    }

    #[test]
    fn eval_when_and() {
        let ctx = KeyContext {
            chat: true,
            streaming: true,
            ..KeyContext::default()
        };
        assert!(eval_when("chat", &ctx));
        assert!(!eval_when("chat && !streaming", &ctx));
        assert!(eval_when("chat && streaming", &ctx));
    }

    #[test]
    fn eval_when_unknown_tag_is_false() {
        let ctx = KeyContext::default();
        assert!(!eval_when("unknownTag", &ctx));
        assert!(eval_when("!unknownTag", &ctx));
    }

    #[test]
    fn default_keybindings_serialize_roundtrip() {
        let defs = default_keybindings();
        let json = serde_json::to_string(&defs).unwrap();
        let parsed: Keybindings = serde_json::from_str(&json).unwrap();
        assert_eq!(defs, parsed);
    }

    #[test]
    fn match_tui_finds_ctrl_c() {
        let kb = default_keybindings();
        let stroke = Keystroke::parse("ctrl+c").unwrap();
        let entry = kb.match_tui(&stroke, &KeyContext::default());
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().command, "stealcode.quit");
    }

    #[test]
    fn match_tui_respects_when() {
        let kb = default_keybindings();
        let stroke = Keystroke::parse("enter").unwrap();

        // In chat without streaming → sendChat
        let ctx_chat = KeyContext {
            chat: true,
            streaming: false,
            ..KeyContext::default()
        };
        let entry = kb.match_tui(&stroke, &ctx_chat);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().command, "stealcode.sendChat");

        // In chat with streaming → noop binding matches (swallows Enter)
        let ctx_streaming = KeyContext {
            chat: true,
            streaming: true,
            ..KeyContext::default()
        };
        let entry = kb.match_tui(&stroke, &ctx_streaming);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().command, "stealcode.noop");
    }

    #[test]
    fn match_tui_palette_binding_only_when_open() {
        let kb = default_keybindings();
        let stroke = Keystroke::parse("up").unwrap();

        let ctx_open = KeyContext {
            palette_open: true,
            ..KeyContext::default()
        };
        let entry = kb.match_tui(&stroke, &ctx_open);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().command, "stealcode.paletteUp");

        let ctx_closed = KeyContext::default();
        let entry = kb.match_tui(&stroke, &ctx_closed);
        assert!(entry.is_none());
    }

    #[test]
    fn default_keybindings_has_expected_commands() {
        let defs = default_keybindings();
        let commands: Vec<&str> =
            defs.tui.iter().map(|e| e.command.as_str()).collect();
        assert!(commands.contains(&"stealcode.quit"));
        assert!(commands.contains(&"stealcode.togglePalette"));
        assert!(commands.contains(&"stealcode.sendChat"));
        assert!(commands.contains(&"stealcode.paletteConfirm"));
    }
}
