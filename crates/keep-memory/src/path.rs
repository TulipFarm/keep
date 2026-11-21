//! Path policy and category inference.
//!
//! Paths are rooted POSIX-style logical names ending in `.md`. On top of the
//! shared [`MemoryPath`] checks this module rejects names that would not
//! survive a round trip through common filesystems, and derives the default
//! category and project from the conventional layout:
//!
//! | Path | Category |
//! | --- | --- |
//! | `/profile.md` | `you` |
//! | `/you/<slug>.md` | `you` |
//! | `/topics/<slug>.md` | `topics` |
//! | `/areas/<slug>.md` | `areas` |
//! | `/people/<slug>.md` | `people` |
//! | `/projects/<uuid>/<name>.md` | `projects`, with the project id |
//! | anything else | uncategorised ("Other") |

use keep_core::memory::MemoryPath;

use crate::error::{LimitKind, MemoryError, Result};

pub const MAX_PATH_BYTES: usize = 200;
pub const MAX_DESCRIPTION_BYTES: usize = 160;

const WINDOWS_RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Validate a path for storage and portable export.
pub fn validate_path(raw: &str) -> Result<MemoryPath> {
    if raw.len() > MAX_PATH_BYTES {
        return Err(MemoryError::Limit {
            kind: LimitKind::PathBytes,
            limit: MAX_PATH_BYTES,
            actual: raw.len(),
        });
    }
    let path = MemoryPath::parse(raw).map_err(|e| MemoryError::InvalidPath(e.to_string()))?;
    for segment in raw[1..].split('/') {
        let stem = segment.split('.').next().unwrap_or("").to_ascii_lowercase();
        if WINDOWS_RESERVED.contains(&stem.as_str()) {
            return Err(MemoryError::InvalidPath(format!(
                "segment {segment:?} is a reserved device name on some platforms"
            )));
        }
        if segment.ends_with(' ') || segment.ends_with('.') || segment.starts_with(' ') {
            return Err(MemoryError::InvalidPath(format!(
                "segment {segment:?} has leading or trailing spaces or dots"
            )));
        }
        if segment.chars().any(|c| "<>\"|?*".contains(c)) {
            return Err(MemoryError::InvalidPath(format!(
                "segment {segment:?} contains a character that is not portable"
            )));
        }
    }
    if raw == "/.md" || raw.ends_with("/.md") {
        return Err(MemoryError::InvalidPath("empty file name".into()));
    }
    Ok(path)
}

/// The key used to detect collisions that would clash on case-insensitive
/// filesystems.
pub fn collision_key(path: &str) -> String {
    path.to_lowercase()
}

pub fn validate_description(description: &str) -> Result<()> {
    if description.len() > MAX_DESCRIPTION_BYTES {
        return Err(MemoryError::Limit {
            kind: LimitKind::DescriptionBytes,
            limit: MAX_DESCRIPTION_BYTES,
            actual: description.len(),
        });
    }
    if description.chars().any(char::is_control) {
        return Err(MemoryError::InvalidRequest(
            "descriptions are one line without control characters".into(),
        ));
    }
    Ok(())
}

/// Default category id and project id for a path. Errors if a `/projects/`
/// path does not name a project UUID.
pub fn infer(path: &MemoryPath) -> Result<(Option<String>, Option<String>)> {
    let segments: Vec<&str> = path.as_str()[1..].split('/').collect();
    let cat = |c: &str| Some(c.to_owned());
    Ok(match segments.as_slice() {
        ["profile.md"] | ["you", _] => (cat("you"), None),
        ["topics", _] => (cat("topics"), None),
        ["areas", _] => (cat("areas"), None),
        ["people", _] => (cat("people"), None),
        ["projects", project, ..] => {
            let id = uuid::Uuid::parse_str(project).map_err(|_| {
                MemoryError::InvalidPath(format!(
                    "project files live under /projects/<uuid>/; {project:?} is not a UUID"
                ))
            })?;
            if segments.len() < 3 {
                return Err(MemoryError::InvalidPath(
                    "project files live under /projects/<uuid>/<name>.md".into(),
                ));
            }
            (cat("projects"), Some(id.hyphenated().to_string()))
        }
        _ => (None, None),
    })
}

/// A readable default display name from the file name.
pub fn default_display_name(path: &MemoryPath) -> String {
    let name = path
        .as_str()
        .rsplit('/')
        .next()
        .unwrap_or("")
        .trim_end_matches(".md");
    let mut out = String::new();
    for (i, word) in name.split(['-', '_']).filter(|w| !w.is_empty()).enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_infers_categories_and_projects() {
        let p = |s: &str| validate_path(s).unwrap();
        assert_eq!(
            infer(&p("/profile.md")).unwrap(),
            (Some("you".into()), None)
        );
        assert_eq!(
            infer(&p("/you/about.md")).unwrap(),
            (Some("you".into()), None)
        );
        assert_eq!(infer(&p("/you/a/b.md")).unwrap(), (None, None));
        assert_eq!(
            infer(&p("/topics/rust.md")).unwrap().0.as_deref(),
            Some("topics")
        );
        assert_eq!(infer(&p("/notes.md")).unwrap(), (None, None));
        let uuid = "6f1c2a52-3b6e-4b0e-9a51-1f3f7e8d2c10";
        assert_eq!(
            infer(&p(&format!("/projects/{uuid}/index.md"))).unwrap(),
            (Some("projects".into()), Some(uuid.into()))
        );
        assert!(infer(&p("/projects/not-a-uuid/index.md")).is_err());
        assert_eq!(
            default_display_name(&p("/topics/ways-of_working.md")),
            "Ways Of Working"
        );
    }

    #[test]
    fn unportable_paths_are_rejected() {
        for bad in [
            "/con.md",
            "/a/aux.md",
            "/a?.md",
            "/x /y.md",
            "/../x.md",
            "/.md",
            "relative.md",
        ] {
            assert!(validate_path(bad).is_err(), "{bad}");
        }
        assert!(matches!(
            validate_path(&format!("/{}.md", "a".repeat(200))),
            Err(MemoryError::Limit {
                kind: LimitKind::PathBytes,
                ..
            })
        ));
        assert_eq!(collision_key("/Topics/A.md"), collision_key("/topics/a.md"));
    }
}
