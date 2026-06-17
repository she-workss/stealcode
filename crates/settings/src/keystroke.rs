use std::{
    borrow::Cow,
    fmt::{self, Display},
};

use serde::{Deserialize, Serialize};
use termina::event::{KeyCode, Modifiers as TerminaModifiers};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidKeystroke {
    pub source: String,
}

impl Display for InvalidKeystroke {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid keystroke '{}': expected modifiers (+ or - separated) followed by a key",
            self.source
        )
    }
}

impl std::error::Error for InvalidKeystroke {}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize,
)]
pub struct Modifiers {
    #[serde(default)]
    pub ctrl: bool,
    #[serde(default)]
    pub alt: bool,
    #[serde(default)]
    pub shift: bool,
    #[serde(default)]
    pub cmd: bool,
}

impl Modifiers {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        !(self.ctrl || self.alt || self.shift || self.cmd)
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize,
)]
pub struct Keystroke {
    pub modifiers: Modifiers,
    pub key: Cow<'static, str>,
}

impl Keystroke {
    pub fn parse(source: &str) -> Result<Self, InvalidKeystroke> {
        let source = source.trim();
        if source.is_empty() {
            return Err(InvalidKeystroke {
                source: source.to_string(),
            });
        }

        let mut modifiers = Modifiers::default();
        let mut key: Option<Cow<'static, str>> = None;

        let parts: Vec<&str> = source.split(['+', '-']).collect();

        for (idx, part) in parts.iter().enumerate() {
            let part = part.trim();
            if part.is_empty() {
                if idx == parts.len() - 1 && parts.len() > 1 {
                    key = Some(Cow::Borrowed("-"));
                }
                continue;
            }

            let lowered = part.to_ascii_lowercase();
            let is_last = idx == parts.len() - 1;
            let is_modifier = matches!(
                lowered.as_str(),
                "ctrl"
                    | "control"
                    | "alt"
                    | "shift"
                    | "cmd"
                    | "command"
                    | "super"
                    | "win"
            );

            if is_modifier && !is_last {
                match lowered.as_str() {
                    "ctrl" | "control" => modifiers.ctrl = true,
                    "alt" => modifiers.alt = true,
                    "shift" => modifiers.shift = true,
                    "cmd" | "command" | "super" | "win" => modifiers.cmd = true,
                    _ => {}
                }
            } else {
                key = Some(normalize_key(&lowered));
            }
        }

        let key = key.ok_or_else(|| InvalidKeystroke {
            source: source.to_string(),
        })?;

        Ok(Self { modifiers, key })
    }

    #[must_use]
    pub fn from_termina(event: &termina::event::KeyEvent) -> Self {
        let m = event.modifiers;
        let modifiers = Modifiers {
            ctrl: m.contains(TerminaModifiers::CONTROL),
            alt: m.contains(TerminaModifiers::ALT),
            shift: m.contains(TerminaModifiers::SHIFT),
            cmd: m.contains(TerminaModifiers::SUPER),
        };
        let key = match event.code {
            KeyCode::Char(c) => normalize_char_key(c, modifiers.shift),
            KeyCode::Function(n) => Cow::Owned(format!("f{n}")),
            KeyCode::Up => Cow::Borrowed("up"),
            KeyCode::Down => Cow::Borrowed("down"),
            KeyCode::Left => Cow::Borrowed("left"),
            KeyCode::Right => Cow::Borrowed("right"),
            KeyCode::Home => Cow::Borrowed("home"),
            KeyCode::End => Cow::Borrowed("end"),
            KeyCode::PageUp => Cow::Borrowed("pageup"),
            KeyCode::PageDown => Cow::Borrowed("pagedown"),
            KeyCode::Insert => Cow::Borrowed("insert"),
            KeyCode::Delete => Cow::Borrowed("delete"),
            KeyCode::Backspace => Cow::Borrowed("backspace"),
            KeyCode::Enter => Cow::Borrowed("enter"),
            KeyCode::Tab => Cow::Borrowed("tab"),
            KeyCode::BackTab => Cow::Borrowed("backtab"),
            KeyCode::Escape => Cow::Borrowed("escape"),
            KeyCode::CapsLock => Cow::Borrowed("capslock"),
            KeyCode::ScrollLock => Cow::Borrowed("scrolllock"),
            KeyCode::NumLock => Cow::Borrowed("numlock"),
            KeyCode::PrintScreen => Cow::Borrowed("printscreen"),
            KeyCode::Pause => Cow::Borrowed("pause"),
            KeyCode::Menu => Cow::Borrowed("menu"),
            KeyCode::KeypadBegin => Cow::Borrowed("keypadbegin"),
            KeyCode::Null => Cow::Borrowed("null"),
            KeyCode::Modifier(_) => Cow::Borrowed("modifier"),
            KeyCode::Media(_) => Cow::Borrowed("media"),
        };
        Self { modifiers, key }
    }

    #[must_use]
    pub fn unparse(&self) -> String {
        let mut out = String::new();
        if self.modifiers.ctrl {
            out.push_str("ctrl-");
        }
        if self.modifiers.alt {
            out.push_str("alt-");
        }
        if self.modifiers.shift {
            out.push_str("shift-");
        }
        if self.modifiers.cmd {
            out.push_str("cmd-");
        }
        out.push_str(&self.key);
        out
    }
}

impl Display for Keystroke {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.unparse())
    }
}

fn normalize_key(key: &str) -> Cow<'static, str> {
    match key {
        "return" | "cr" => Cow::Borrowed("enter"),
        "del" => Cow::Borrowed("delete"),
        "esc" => Cow::Borrowed("escape"),
        "pgup" => Cow::Borrowed("pageup"),
        "pgdown" | "pgdn" => Cow::Borrowed("pagedown"),
        "space" | "spc" => Cow::Borrowed(" "),
        "bs" => Cow::Borrowed("backspace"),
        _ => Cow::Owned(key.to_string()),
    }
}

fn normalize_char_key(c: char, shift: bool) -> Cow<'static, str> {
    if c == ' ' {
        return Cow::Borrowed(" ");
    }
    if shift && c.is_ascii_uppercase() {
        return Cow::Owned(c.to_ascii_lowercase().to_string());
    }
    Cow::Owned(c.to_string())
}

#[cfg(test)]
mod tests {
    use termina::event::{KeyEvent, KeyEventKind, KeyEventState};

    use super::*;

    #[test]
    fn parse_simple_key() {
        let k = Keystroke::parse("a").unwrap();
        assert!(k.modifiers.is_empty());
        assert_eq!(k.key, "a");
    }

    #[test]
    fn parse_with_plus_separator() {
        let k = Keystroke::parse("ctrl+enter").unwrap();
        assert!(k.modifiers.ctrl);
        assert!(!k.modifiers.shift);
        assert_eq!(k.key, "enter");
    }

    #[test]
    fn parse_with_dash_separator() {
        let k = Keystroke::parse("shift-tab").unwrap();
        assert!(k.modifiers.shift);
        assert_eq!(k.key, "tab");
    }

    #[test]
    fn parse_multiple_modifiers() {
        let k = Keystroke::parse("ctrl+shift+p").unwrap();
        assert!(k.modifiers.ctrl);
        assert!(k.modifiers.shift);
        assert_eq!(k.key, "p");
    }

    #[test]
    fn parse_function_key() {
        let k = Keystroke::parse("f12").unwrap();
        assert!(k.modifiers.is_empty());
        assert_eq!(k.key, "f12");
    }

    #[test]
    fn parse_cmd_alias() {
        let k = Keystroke::parse("cmd+s").unwrap();
        assert!(k.modifiers.cmd);
        assert_eq!(k.key, "s");
    }

    #[test]
    fn parse_win_alias() {
        let k = Keystroke::parse("win+e").unwrap();
        assert!(k.modifiers.cmd);
        assert_eq!(k.key, "e");
    }

    #[test]
    fn parse_super_alias() {
        let k = Keystroke::parse("super+1").unwrap();
        assert!(k.modifiers.cmd);
        assert_eq!(k.key, "1");
    }

    #[test]
    fn parse_trailing_dash_as_minus_key() {
        let k = Keystroke::parse("ctrl+-").unwrap();
        assert!(k.modifiers.ctrl);
        assert_eq!(k.key, "-");
    }

    #[test]
    fn parse_empty_fails() {
        assert!(Keystroke::parse("").is_err());
    }

    #[test]
    fn parse_whitespace_only_fails() {
        assert!(Keystroke::parse("   ").is_err());
    }

    #[test]
    fn unparse_roundtrip() {
        let k = Keystroke::parse("ctrl+shift+enter").unwrap();
        assert_eq!(k.unparse(), "ctrl-shift-enter");
    }

    #[test]
    fn from_termina_ctrl_c() {
        use termina::event::{KeyEvent, KeyEventKind, KeyEventState};
        let event = KeyEvent {
            code: KeyCode::Char('c'),
            modifiers: TerminaModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        };
        let k = Keystroke::from_termina(&event);
        assert!(k.modifiers.ctrl);
        assert_eq!(k.key, "c");
    }

    #[test]
    fn from_termina_shift_a() {
        use termina::event::{KeyEvent, KeyEventKind, KeyEventState};
        let event = KeyEvent {
            code: KeyCode::Char('A'),
            modifiers: TerminaModifiers::SHIFT,
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        };
        let k = Keystroke::from_termina(&event);
        assert!(k.modifiers.shift);
        assert_eq!(k.key, "a");
    }

    #[test]
    fn from_termina_f12() {
        let event = KeyEvent {
            code: KeyCode::Function(12),
            modifiers: TerminaModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        let k = Keystroke::from_termina(&event);
        assert!(k.modifiers.is_empty());
        assert_eq!(k.key, "f12");
    }
}
