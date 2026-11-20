//! Conservative source-time annotations. Old extraction parsers stay unchanged.

use chrono::{Datelike, Duration, Months, NaiveDate};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelativeTime {
    pub phrase: String,
    pub session_date: NaiveDate,
    pub resolved: String,
}

pub fn resolve(phrase: &str, anchor: NaiveDate) -> Option<RelativeTime> {
    let lower = phrase.to_lowercase();
    let words: Vec<_> = lower.split_whitespace().collect();
    let resolved = match words.as_slice() {
        ["next" | "last" | "this", "month"] => {
            let d = match words[0] {
                "next" => anchor.checked_add_months(Months::new(1))?,
                "last" => anchor.checked_sub_months(Months::new(1))?,
                _ => anchor,
            };
            d.format("%B %Y").to_string()
        }
        ["next" | "last" | "this", "year"] => (anchor.year()
            + match words[0] {
                "next" => 1,
                "last" => -1,
                _ => 0,
            })
        .to_string(),
        ["last" | "next" | "this", "week" | "weekend"] => {
            format!("{} anchored to {anchor}", lower)
        }
        [n, unit, "ago"] if matches!(*unit, "weeks" | "weekends" | "months" | "years") => {
            let n = crate::session_extract::number_of(n)?;
            if !(0..=200).contains(&n) {
                return None;
            }
            match *unit {
                "years" => format!(
                    "{lower} as of {anchor} (approximately {})",
                    anchor.year() - n as i32
                ),
                "months" => format!(
                    "{lower} as of {anchor} ({})",
                    anchor
                        .checked_sub_months(Months::new(n as u32))?
                        .format("%B %Y")
                ),
                _ => format!("{n} {unit} before {anchor}"),
            }
        }
        ["tomorrow"] => anchor.checked_add_signed(Duration::days(1))?.to_string(),
        ["yesterday"] => anchor.checked_sub_signed(Duration::days(1))?.to_string(),
        ["today"] => anchor.to_string(),
        ["last", day] if crate::session_extract::weekday_of(day).is_some() => {
            crate::session_extract::resolve_when(&lower, anchor)?.to_string()
        }
        [n, "day" | "days", "ago"] => {
            let n = crate::session_extract::number_of(n)?;
            if !(0..=36600).contains(&n) {
                return None;
            }
            anchor.checked_sub_signed(Duration::days(n))?.to_string()
        }
        _ => return None,
    };
    Some(RelativeTime {
        phrase: phrase.into(),
        session_date: anchor,
        resolved,
    })
}

pub fn in_text(text: &str, anchor: NaiveDate) -> Vec<RelativeTime> {
    let words: Vec<_> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let found = (1..=3.min(words.len() - i))
            .rev()
            .find_map(|n| resolve(&words[i..i + n].join(" "), anchor).map(|r| (n, r)));
        if let Some((n, r)) = found {
            if !out.contains(&r) {
                out.push(r);
            }
            i += n;
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn month_rollover_and_imprecise_week_do_not_invent_days() {
        let anchor = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
        assert_eq!(
            resolve("next month", anchor).unwrap().resolved,
            "January 2025"
        );
        assert_eq!(
            resolve("two weekends ago", anchor).unwrap().resolved,
            "2 weekends before 2024-12-31"
        );
        assert!(resolve("sometime soon", anchor).is_none());
    }

    #[test]
    fn source_age_does_not_use_wall_clock_and_keeps_phrase() {
        let anchor = NaiveDate::from_ymd_opt(2023, 5, 6).unwrap();
        let refs = in_text("I moved ten years ago. I'll visit next month.", anchor);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].phrase, "ten years ago");
        assert!(refs[0].resolved.contains("2013"));
        assert_eq!(refs[1].resolved, "June 2023");
        assert_eq!(refs[0].session_date, anchor);
    }
}
