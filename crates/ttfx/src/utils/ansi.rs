//! ANSI escape sequences, ported from utils/ansitools.py + utils/colorterm.py.

pub const DEC_SAVE_CURSOR: &str = "\x1b7";
pub const DEC_RESTORE_CURSOR: &str = "\x1b8";
pub const HIDE_CURSOR: &str = "\x1b[?25l";
pub const SHOW_CURSOR: &str = "\x1b[?25h";
pub const RESET_ALL: &str = "\x1b[0m";
pub const CLEAR_TO_END_OF_SCREEN: &str = "\x1b[0J";
pub const BOLD: &str = "\x1b[1m";
pub const ITALIC: &str = "\x1b[3m";
pub const UNDERLINE: &str = "\x1b[4m";
pub const BLINK: &str = "\x1b[5m";
pub const REVERSE: &str = "\x1b[7m";
pub const HIDDEN: &str = "\x1b[8m";
pub const STRIKETHROUGH: &str = "\x1b[9m";

pub fn move_cursor_up(y: usize) -> String {
    format!("\x1b[{y}A")
}

/// A resolved color code ready for SGR emission: hex string => 24-bit, int =>
/// 8-bit. Mirrors the str|int union threaded through colorterm/animation
/// upstream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ColorCode {
    Rgb(String), // hex without '#', case preserved as upstream passes it
    Xterm(u8),
}

/// Decimal digits of a byte, without going through core::fmt. Every restyled
/// character reassembles its SGR sequence, so the formatting machinery shows up
/// in profiles.
#[inline]
fn push_decimal(out: &mut String, value: u8) {
    if value >= 100 {
        out.push((b'0' + value / 100) as char);
    }
    if value >= 10 {
        out.push((b'0' + (value / 10) % 10) as char);
    }
    out.push((b'0' + value % 10) as char);
}

/// colorterm._color: fg selector 38, bg selector 48.
fn sgr_color(code: &ColorCode, location: u8, out: &mut String) {
    out.push_str("\x1b[");
    push_decimal(out, location);
    match code {
        ColorCode::Rgb(hex) => {
            let s = hex.trim_matches('#');
            let r = u8::from_str_radix(&s[0..2], 16).unwrap();
            let g = u8::from_str_radix(&s[2..4], 16).unwrap();
            let b = u8::from_str_radix(&s[4..6], 16).unwrap();
            out.push_str(";2;");
            push_decimal(out, r);
            out.push(';');
            push_decimal(out, g);
            out.push(';');
            push_decimal(out, b);
        }
        ColorCode::Xterm(n) => {
            out.push_str(";5;");
            push_decimal(out, *n);
        }
    }
    out.push('m');
}

pub fn fg(code: &ColorCode, out: &mut String) {
    sgr_color(code, 38, out);
}

pub fn bg(code: &ColorCode, out: &mut String) {
    sgr_color(code, 48, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgr_emission() {
        let mut s = String::new();
        fg(&ColorCode::Rgb("ff0080".into()), &mut s);
        assert_eq!(s, "\x1b[38;2;255;0;128m");
        s.clear();
        bg(&ColorCode::Xterm(42), &mut s);
        assert_eq!(s, "\x1b[48;5;42m");
    }
}
