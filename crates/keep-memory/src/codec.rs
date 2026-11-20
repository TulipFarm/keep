//! Lossless Markdown codec.
//!
//! A file's visible Markdown is cut into contiguous blocks that cover every
//! byte exactly once. Blocks become entries; rendering concatenates their
//! fragments, so `render(parse(text)) == text` byte for byte for any admitted
//! input. Nothing is normalised: line endings, trailing spaces, blank lines,
//! front matter, HTML and unsupported syntax all stay as they were.
//!
//! The parser understands only what it needs for stable identities:
//! ATX headings (`#` to `######`) outside fenced code, a leading `---` front
//! matter block and, when a file opts in, top-level list items. Everything
//! else is carried as opaque text inside the surrounding block.

use serde::{Deserialize, Serialize};

/// How finely a file is split into entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryMode {
    /// The whole file is one opaque entry (the raw-text baseline).
    Raw,
    /// One entry per heading section; text before the first heading and
    /// front matter are opaque entries. The default.
    #[default]
    Sections,
    /// Sections, plus one entry per top-level list item.
    SectionsAndBullets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Section,
    Bullet,
    Opaque,
}

/// A contiguous byte range of the source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: EntryKind,
    pub start: usize,
    pub end: usize,
    /// A structural key used for alignment: the heading line for sections,
    /// the first line for bullets.
    pub key: Option<String>,
}

/// Why text cannot be admitted.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TextError {
    #[error("text is not valid UTF-8")]
    InvalidUtf8,
    #[error("text contains a NUL byte")]
    Nul,
}

/// Admit raw bytes as Markdown text without altering them.
pub fn admit_bytes(bytes: &[u8]) -> Result<&str, TextError> {
    let text = std::str::from_utf8(bytes).map_err(|_| TextError::InvalidUtf8)?;
    admit(text)?;
    Ok(text)
}

pub fn admit(text: &str) -> Result<(), TextError> {
    if text.contains('\0') {
        return Err(TextError::Nul);
    }
    Ok(())
}

/// Lines with their terminators, plus byte offsets.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut offset = 0;
    text.split_inclusive('\n').map(move |line| {
        let at = offset;
        offset += line.len();
        (at, line)
    })
}

fn strip_eol(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// Up to three leading spaces are allowed before block markers.
fn after_indent(line: &str) -> Option<&str> {
    let spaces = line.bytes().take_while(|b| *b == b' ').count();
    (spaces <= 3).then(|| &line[spaces..])
}

pub(crate) fn heading_level(line: &str) -> Option<usize> {
    let body = after_indent(strip_eol(line))?;
    let hashes = body.bytes().take_while(|b| *b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    match body.as_bytes().get(hashes) {
        None | Some(b' ') | Some(b'\t') => Some(hashes),
        _ => None,
    }
}

/// Tracks fenced code so markers inside code are ignored.
#[derive(Default)]
struct Fence {
    open: Option<(u8, usize)>,
}

impl Fence {
    /// Feed a line; returns true if the line is inside (or delimits) a fence.
    fn feed(&mut self, line: &str) -> bool {
        let body = after_indent(strip_eol(line));
        let marker = body.and_then(|b| {
            let c = *b.as_bytes().first()?;
            if c != b'`' && c != b'~' {
                return None;
            }
            let n = b.bytes().take_while(|x| *x == c).count();
            (n >= 3).then_some((c, n, &b[n..]))
        });
        match (self.open, marker) {
            (None, Some((c, n, rest))) => {
                if c == b'`' && rest.contains('`') {
                    return false;
                }
                self.open = Some((c, n));
                true
            }
            (None, None) => false,
            (Some((c, n)), Some((mc, mn, rest)))
                if mc == c && mn >= n && rest.trim().is_empty() =>
            {
                self.open = None;
                true
            }
            (Some(_), _) => true,
        }
    }
}

fn front_matter_end(text: &str) -> Option<usize> {
    let mut it = lines(text);
    let (_, first) = it.next()?;
    if strip_eol(first) != "---" || !first.ends_with('\n') {
        return None;
    }
    for (at, line) in it {
        let l = strip_eol(line);
        if l == "---" || l == "..." {
            return Some(at + line.len());
        }
    }
    None
}

fn list_marker(line: &str) -> bool {
    let l = strip_eol(line);
    let b = l.as_bytes();
    match b.first() {
        Some(b'-' | b'*' | b'+') => matches!(b.get(1), Some(b' ' | b'\t')),
        Some(c) if c.is_ascii_digit() => {
            let digits = b.iter().take_while(|c| c.is_ascii_digit()).count();
            digits <= 9
                && matches!(b.get(digits), Some(b'.' | b')'))
                && matches!(b.get(digits + 1), Some(b' ' | b'\t'))
        }
        _ => false,
    }
}

/// Split `text` into blocks that cover it exactly.
pub fn parse(text: &str, mode: EntryMode) -> Vec<Block> {
    if text.is_empty() {
        return Vec::new();
    }
    if mode == EntryMode::Raw {
        return vec![Block {
            kind: EntryKind::Opaque,
            start: 0,
            end: text.len(),
            key: None,
        }];
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    if let Some(end) = front_matter_end(text) {
        blocks.push(Block {
            kind: EntryKind::Opaque,
            start: 0,
            end,
            key: Some("front-matter".into()),
        });
        start = end;
    }
    let mut cur_start = start;
    let mut cur_key: Option<String> = None;
    let mut fence = Fence::default();
    for (at, line) in lines(&text[start..]).map(|(a, l)| (a + start, l)) {
        if fence.feed(line) {
            continue;
        }
        if heading_level(line).is_some() {
            if at > cur_start {
                blocks.push(section_or_opaque(cur_start, at, cur_key.take()));
            }
            cur_start = at;
            cur_key = Some(strip_eol(line).trim().to_owned());
        }
    }
    if text.len() > cur_start {
        blocks.push(section_or_opaque(cur_start, text.len(), cur_key));
    }
    if mode == EntryMode::SectionsAndBullets {
        blocks = blocks
            .into_iter()
            .flat_map(|b| split_bullets(text, b))
            .collect();
    }
    blocks
}

fn section_or_opaque(start: usize, end: usize, key: Option<String>) -> Block {
    Block {
        kind: if key.is_some() {
            EntryKind::Section
        } else {
            EntryKind::Opaque
        },
        start,
        end,
        key,
    }
}

/// Split one section or opaque block into its lead-in, top-level list items
/// and any trailing text.
fn split_bullets(text: &str, block: Block) -> Vec<Block> {
    if block.key.as_deref() == Some("front-matter") {
        return vec![block];
    }
    let body = &text[block.start..block.end];
    let mut out: Vec<Block> = Vec::new();
    let mut seg_start = block.start;
    let mut seg_kind = block.kind;
    let mut seg_key = block.key.clone();
    // Whether the current segment is a list item still accepting
    // continuation lines.
    let mut in_item = false;
    let mut fence = Fence::default();
    for (i, (rel, line)) in lines(body).enumerate() {
        let at = block.start + rel;
        if i == 0 && block.kind == EntryKind::Section {
            continue; // the heading line itself
        }
        let inside_fence = fence.open.is_some();
        let opens_fence = fence.feed(line) && !inside_fence;
        if inside_fence {
            continue;
        }
        let blank = strip_eol(line).trim().is_empty();
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if in_item && (blank || indented) {
            continue; // continuation of the item, including indented code
        }
        if !opens_fence && list_marker(line) {
            push(&mut out, seg_start, at, seg_kind, seg_key.take());
            seg_start = at;
            seg_kind = EntryKind::Bullet;
            seg_key = Some(strip_eol(line).trim().to_owned());
            in_item = true;
        } else if seg_kind == EntryKind::Bullet {
            push(&mut out, seg_start, at, seg_kind, seg_key.take());
            seg_start = at;
            seg_kind = EntryKind::Opaque;
            in_item = false;
        }
    }
    push(&mut out, seg_start, block.end, seg_kind, seg_key);
    out
}

fn push(out: &mut Vec<Block>, start: usize, end: usize, kind: EntryKind, key: Option<String>) {
    if end > start {
        out.push(Block {
            kind,
            start,
            end,
            key,
        });
    }
}

/// Render fragments back to text. Exists so callers never reconstruct text
/// any other way.
pub fn render<'a>(fragments: impl IntoIterator<Item = &'a str>) -> String {
    fragments.into_iter().collect()
}

/// Paragraph-sized units inside a block, used for excerpts: runs of
/// non-blank lines together with the blank lines that follow them. Fenced
/// code stays in one unit.
pub fn paragraph_units(text: &str) -> Vec<(usize, usize)> {
    let mut units = Vec::new();
    let mut start = 0;
    let mut seen_text = false;
    let mut fence = Fence::default();
    for (at, line) in lines(text) {
        let fenced_before = fence.open.is_some();
        let fenced = fence.feed(line);
        let blank = strip_eol(line).trim().is_empty();
        if !fenced_before && !fenced && !blank && seen_text && at > start {
            // A new paragraph begins after a blank line.
            let prev_blank = text[..at]
                .strip_suffix('\n')
                .map(|p| {
                    let last = p.rsplit('\n').next().unwrap_or("");
                    strip_eol(last).trim().is_empty()
                })
                .unwrap_or(false);
            if prev_blank || heading_level(line).is_some() {
                units.push((start, at));
                start = at;
            }
        }
        if !blank {
            seen_text = true;
        }
    }
    if text.len() > start {
        units.push((start, text.len()));
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn kinds(text: &str, mode: EntryMode) -> Vec<(EntryKind, String)> {
        parse(text, mode)
            .into_iter()
            .map(|b| (b.kind, text[b.start..b.end].to_owned()))
            .collect()
    }

    fn assert_cover(text: &str, mode: EntryMode) {
        let blocks = parse(text, mode);
        let mut at = 0;
        for b in &blocks {
            assert_eq!(b.start, at, "contiguous");
            assert!(b.end > b.start, "non-empty");
            assert!(text.is_char_boundary(b.start) && text.is_char_boundary(b.end));
            at = b.end;
        }
        assert_eq!(at, text.len());
        assert_eq!(
            render(blocks.iter().map(|b| &text[b.start..b.end])),
            text,
            "lossless"
        );
    }

    #[test]
    fn sections_split_at_headings_outside_code() {
        let text =
            "---\ntitle: x\n---\nIntro\n# One\nbody\n```\n# not a heading\n```\n## Two\r\nmore\n";
        let k = kinds(text, EntryMode::Sections);
        assert_eq!(k.len(), 4);
        assert_eq!(k[0], (EntryKind::Opaque, "---\ntitle: x\n---\n".into()));
        assert_eq!(k[1], (EntryKind::Opaque, "Intro\n".into()));
        assert_eq!(k[2].0, EntryKind::Section);
        assert!(k[2].1.contains("# not a heading"));
        assert_eq!(k[3], (EntryKind::Section, "## Two\r\nmore\n".into()));
        assert_cover(text, EntryMode::Sections);
    }

    #[test]
    fn hashes_without_space_are_not_headings() {
        assert_eq!(parse("#tag\ntext", EntryMode::Sections).len(), 1);
        assert_eq!(
            parse("    # indented code\n", EntryMode::Sections)[0].kind,
            EntryKind::Opaque
        );
    }

    #[test]
    fn opted_in_lists_get_bullet_entries() {
        let text = "## Prefs\nLead.\n- tea\n  with milk\n\n- coffee\n1. first\nAfter text\n```\n- in code\n```\n";
        let k = kinds(text, EntryMode::SectionsAndBullets);
        let summary: Vec<_> = k.iter().map(|(k, t)| (*k, t.as_str())).collect();
        assert_eq!(
            summary,
            vec![
                (EntryKind::Section, "## Prefs\nLead.\n"),
                (EntryKind::Bullet, "- tea\n  with milk\n\n"),
                (EntryKind::Bullet, "- coffee\n"),
                (EntryKind::Bullet, "1. first\n"),
                (EntryKind::Opaque, "After text\n```\n- in code\n```\n"),
            ]
        );
        assert_cover(text, EntryMode::SectionsAndBullets);
    }

    #[test]
    fn raw_mode_is_one_block_and_empty_text_has_none() {
        assert_eq!(parse("# a\n# b\n", EntryMode::Raw).len(), 1);
        assert!(parse("", EntryMode::Sections).is_empty());
    }

    #[test]
    fn admission_rejects_nul_and_invalid_utf8() {
        assert_eq!(admit_bytes(b"a\0b"), Err(TextError::Nul));
        assert_eq!(admit_bytes(&[0xff, 0x00]), Err(TextError::InvalidUtf8));
        assert_eq!(admit_bytes("é\r\n".as_bytes()), Ok("é\r\n"));
    }

    #[test]
    fn paragraph_units_cover_text() {
        let text = "# H\npara one\nline two\n\npara two\n\n```\na\n\nb\n```\n";
        let units = paragraph_units(text);
        assert_eq!(units.first().unwrap().0, 0);
        assert_eq!(units.last().unwrap().1, text.len());
        for w in units.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        assert!(units.iter().any(|(s, e)| text[*s..*e].contains("a\n\nb")));
    }

    fn markdownish() -> impl Strategy<Value = String> {
        let piece = prop_oneof![
            Just("# ".to_owned()),
            Just("## ".to_owned()),
            Just("#".to_owned()),
            Just("- ".to_owned()),
            Just("1. ".to_owned()),
            Just("```".to_owned()),
            Just("~~~".to_owned()),
            Just("---".to_owned()),
            Just("\n".to_owned()),
            Just("\r\n".to_owned()),
            Just("\r".to_owned()),
            Just("  ".to_owned()),
            Just("\t".to_owned()),
            Just("<b>".to_owned()),
            Just("[l](x.md)".to_owned()),
            "[a-zé漢🙂 ]{0,6}",
        ];
        proptest::collection::vec(piece, 0..60).prop_map(|v| v.concat())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn round_trip_is_byte_exact(text in markdownish()) {
            for mode in [EntryMode::Raw, EntryMode::Sections, EntryMode::SectionsAndBullets] {
                assert_cover(&text, mode);
            }
            let units = paragraph_units(&text);
            prop_assert_eq!(render(units.iter().map(|(s, e)| &text[*s..*e])), text.clone());
        }

        #[test]
        fn arbitrary_unicode_round_trips(text in "\\PC{0,200}") {
            for mode in [EntryMode::Sections, EntryMode::SectionsAndBullets] {
                assert_cover(&text, mode);
            }
        }
    }
}
