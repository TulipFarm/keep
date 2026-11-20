//! Split episode text into retrievable chunks with exact character spans.

/// A half-open character range `[start, end)` into the episode text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Default upper bound on chunk length in characters.
pub const DEFAULT_CHUNK_CHARS: usize = 1_200;

/// Greedy sentence packing: sentences are appended until the next one would
/// exceed `max_chars`; a single over-long sentence is hard-split. Offsets are
/// in characters (not bytes) so they survive any storage encoding. Short
/// texts produce one chunk equal to the trimmed input.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<Span> {
    let max_chars = max_chars.max(1);
    let chars: Vec<char> = text.chars().collect();
    let mut sentences: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for (i, c) in chars.iter().enumerate() {
        let boundary = matches!(c, '.' | '!' | '?' | '\n')
            && chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if boundary {
            sentences.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < chars.len() {
        sentences.push((start, chars.len()));
    }

    let mut out = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    let flush = |s: usize, e: usize, out: &mut Vec<Span>| {
        let (s, e) = trim(&chars, s, e);
        if s < e {
            out.push(Span {
                start: s,
                end: e,
                text: chars[s..e].iter().collect(),
            });
        }
    };
    for (s, e) in sentences {
        match cur {
            Some((cs, _)) if e - cs <= max_chars => cur = Some((cs, e)),
            Some((cs, ce)) => {
                flush(cs, ce, &mut out);
                cur = Some((s, e));
            }
            None => cur = Some((s, e)),
        }
        while let Some((cs, ce)) = cur {
            if ce - cs <= max_chars {
                break;
            }
            flush(cs, cs + max_chars, &mut out);
            cur = Some((cs + max_chars, ce));
        }
    }
    if let Some((cs, ce)) = cur {
        flush(cs, ce, &mut out);
    }
    out
}

fn trim(chars: &[char], mut s: usize, mut e: usize) -> (usize, usize) {
    while s < e && chars[s].is_whitespace() {
        s += 1;
    }
    while e > s && chars[e - 1].is_whitespace() {
        e -= 1;
    }
    (s, e)
}

/// Character offset of `needle` in `hay`, if present.
pub fn char_find(hay: &str, needle: &str) -> Option<(usize, usize)> {
    let byte = hay.find(needle)?;
    let start = hay[..byte].chars().count();
    Some((start, start + needle.chars().count()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_point_back_into_the_original_text() {
        let text = "Ünïcode first. Second sentence here! Third?  Fourth";
        let chunks = chunk_text(text, 20);
        let chars: Vec<char> = text.chars().collect();
        assert!(chunks.len() >= 3);
        for c in &chunks {
            let back: String = chars[c.start..c.end].iter().collect();
            assert_eq!(back, c.text);
            assert!(c.text.chars().count() <= 20);
        }
        assert_eq!(chunks[0].text, "Ünïcode first.");
    }

    #[test]
    fn short_text_is_one_chunk_and_long_sentences_split() {
        assert_eq!(chunk_text("  hi there  ", 100).len(), 1);
        assert_eq!(chunk_text("", 100).len(), 0);
        let long = "a".repeat(25);
        let parts = chunk_text(&long, 10);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2].text.len(), 5);
    }

    #[test]
    fn char_find_counts_characters() {
        assert_eq!(char_find("é Lisbon", "Lisbon"), Some((2, 8)));
        assert_eq!(char_find("abc", "z"), None);
    }
}
