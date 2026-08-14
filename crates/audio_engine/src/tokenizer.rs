//! SentencePiece tokenizer decode:
//!   * U+2581 ("▁", UTF-8 E2 96 81) -> ASCII space, byte-for-byte.
//!   * byte-fallback pieces "<0xHH>" -> single byte.
//!   * multilingual language-tag pieces "<ll-RR>" are stripped from the public
//!     transcript by default (they are emitted per segment to mark the
//!     language; `is_lang_tag_piece` in the reference).
//!   * runs of ASCII spaces collapse to one; both ends trimmed.

use std::borrow::Cow;

use crate::nemotron::config::ModelConfig;

/// U+2581 "▁" in UTF-8.
const SP_MARKER: &[u8] = &[0xE2, 0x96, 0x81];

#[derive(Debug)]
pub struct Tokenizer {
    pub tokens: Vec<String>,
}

impl Tokenizer {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            tokens: cfg.tokens.clone(),
        }
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// SentencePiece decode over a token id sequence.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let Some(p) = self.piece(id) else { continue };
            if let Some(byte) = byte_fallback(p) {
                out.push(byte);
                continue;
            }
            let bytes = p.as_bytes();
            let mut j = 0usize;
            while j < bytes.len() {
                if j + SP_MARKER.len() <= bytes.len()
                    && &bytes[j..j + SP_MARKER.len()] == SP_MARKER
                {
                    out.push(b' ');
                    j += SP_MARKER.len();
                } else {
                    out.push(bytes[j]);
                    j += 1;
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Language-tag piece check (mirrors `is_lang_tag_piece`): "<ll-RR>"
    /// with a 2-3 lowercase language and 2-4 alphanumeric region.
    pub fn is_lang_tag_piece(&self, piece: &str) -> bool {
        let b = piece.as_bytes();
        let n = b.len();
        if n < 7 || b[0] != b'<' || b[n - 1] != b'>' {
            return false;
        }
        let end = n - 1;
        let mut i = 1;
        let lang0 = i;
        while i < end && b[i].is_ascii_lowercase() {
            i += 1;
        }
        let lang_len = i - lang0;
        if lang_len < 2 || lang_len > 3 {
            return false;
        }
        if i >= end || b[i] != b'-' {
            return false;
        }
        i += 1;
        let reg0 = i;
        while i < end && b[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let reg_len = i - reg0;
        if reg_len < 2 || reg_len > 4 {
            return false;
        }
        i == end
    }

    /// Is this token dropped from the public transcript by default?
    pub fn is_strippable(&self, id: u32) -> bool {
        self.piece(id).map_or(false, |p| self.is_lang_tag_piece(p))
    }

    /// Decode a transcript: drop language tags, SPM-decode, collapse
    /// whitespace runs and trim (mirrors `decode_and_populate`).
    pub fn decode_transcript(&self, ids: &[u32], strip_tags: bool) -> String {
        let filtered: Cow<'_, [u32]> = if strip_tags {
            Cow::Owned(
                ids.iter()
                    .copied()
                    .filter(|&id| !self.is_strippable(id))
                    .collect(),
            )
        } else {
            Cow::Borrowed(ids)
        };
        let mut text = self.decode(&filtered);
        normalize_transcript_whitespace(&mut text);
        text
    }
}

/// "<0xHH>" (exactly six chars) -> byte value, else -1.
fn byte_fallback(p: &str) -> Option<u8> {
    let b = p.as_bytes();
    if b.len() != 6
        || b[0] != b'<'
        || b[1] != b'0'
        || b[2] != b'x'
        || b[5] != b'>'
    {
        return None;
    }
    let hi = hex_nibble(b[3])?;
    let lo = hex_nibble(b[4])?;
    Some((hi << 4) | lo)
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'F' => Some(10 + c - b'A'),
        b'a'..=b'f' => Some(10 + c - b'a'),
        _ => None,
    }
}

/// Collapse runs of ASCII spaces to one, trim both ends.
pub fn normalize_transcript_whitespace(s: &mut String) {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        let is_space = ch == ' ';
        if is_space && prev_space {
            continue;
        }
        out.push(ch);
        prev_space = is_space;
    }
    let end = out.trim_end().len();
    out.truncate(end);
    let start = out.trim_start().len();
    out.drain(..end - start);
    *s = out;
}

#[cfg(test)]
mod tests {
    use super::normalize_transcript_whitespace;

    #[test]
    fn trims_ascii() {
        let mut s = String::from("  hello  ");
        normalize_transcript_whitespace(&mut s);
        assert_eq!(s, "hello");
    }

    #[test]
    fn trims_around_multibyte() {
        let mut s = String::from(" привет ");
        normalize_transcript_whitespace(&mut s);
        assert_eq!(s, "привет");
    }

    #[test]
    fn collapses_and_trims() {
        let mut s = String::from("a  b   c ");
        normalize_transcript_whitespace(&mut s);
        assert_eq!(s, "a b c");
    }

    #[test]
    fn handles_empty_and_whitespace_only() {
        let mut s = String::from("");
        normalize_transcript_whitespace(&mut s);
        assert_eq!(s, "");
        let mut s = String::from("   ");
        normalize_transcript_whitespace(&mut s);
        assert_eq!(s, "");
    }
}
