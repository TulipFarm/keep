//! The time a question refers to.
//!
//! [`question_window`] reads phrases such as "last month", "in the past
//! three weeks", "two weeks ago", "in May" or "last Saturday" in a question
//! and turns them into a calendar window, counted back from the date the
//! question was asked. Retrieval uses the window to favour facts and
//! evidence dated inside it; it never filters on it, because people are
//! loose with time ("last month" said in early June can mean May or the
//! past few weeks).
//!
//! Parsing is deterministic and conservative: a question without a phrase
//! it recognises has no window.

use chrono::{Datelike, Duration, Months, NaiveDate, Weekday};

use crate::session_extract::{month_of, number_of, previous, weekday_of, year_of};

/// An inclusive range of calendar dates and the phrase that set it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeWindow {
    pub from: NaiveDate,
    pub to: NaiveDate,
    pub phrase: String,
}

impl TimeWindow {
    pub fn contains(&self, d: NaiveDate) -> bool {
        self.from <= d && d <= self.to
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Unit {
    Day,
    Week,
    Month,
    Year,
}

fn unit_of(w: &str) -> Option<Unit> {
    Some(match w.trim_end_matches('s') {
        "day" | "night" => Unit::Day,
        "week" => Unit::Week,
        "month" => Unit::Month,
        "year" => Unit::Year,
        _ => return None,
    })
}

fn back(anchor: NaiveDate, n: i64, unit: Unit) -> Option<NaiveDate> {
    match unit {
        Unit::Day => anchor.checked_sub_signed(Duration::days(n)),
        Unit::Week => anchor.checked_sub_signed(Duration::days(7 * n)),
        Unit::Month => anchor.checked_sub_months(Months::new(u32::try_from(n).ok()?)),
        Unit::Year => anchor.checked_sub_months(Months::new(12 * u32::try_from(n).ok()?)),
    }
}

/// How far either side of "N units ago" still counts.
fn slack(n: i64, unit: Unit) -> Duration {
    match unit {
        Unit::Day => Duration::days(1 + n / 7),
        Unit::Week => Duration::days(4),
        Unit::Month => Duration::days(15),
        Unit::Year => Duration::days(183),
    }
}

fn window(from: NaiveDate, to: NaiveDate, phrase: &[&str]) -> Option<TimeWindow> {
    (from <= to).then(|| TimeWindow {
        from,
        to,
        phrase: phrase.join(" "),
    })
}

fn month_end(y: i32, m: u32) -> Option<NaiveDate> {
    let next = if m == 12 {
        NaiveDate::from_ymd_opt(y + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(y, m + 1, 1)
    }?;
    next.pred_opt()
}

/// A count word at `w[i]`: digits, number words, "a", "a few", "a couple".
fn count_at(w: &[&str], i: usize) -> Option<(i64, usize)> {
    match w.get(i)? {
        &"few" | &"several" => Some((3, i)),
        &"couple" => Some((2, i)),
        &"a" | &"an" => match w.get(i + 1) {
            Some(&"few") => Some((3, i)),
            Some(&"couple") => Some((2, i)),
            _ => Some((1, i)),
        },
        x => number_of(x)
            .filter(|n| (1..=60).contains(n))
            .map(|n| (n, i)),
    }
}

const SEASONS: [(&str, u32, u32); 5] = [
    ("spring", 3, 5),
    ("summer", 6, 8),
    ("fall", 9, 11),
    ("autumn", 9, 11),
    ("winter", 12, 2),
];

/// The window `question` refers to, counted from `asked`, or `None` when
/// it names no time this parser understands.
///
/// Understood, in order of precedence:
/// - `[in|over|during|within] the last|past N days/weeks/months/years`,
///   `last|past N ...`, `the past few months`: from N units back to `asked`;
/// - `N days/weeks/months/years ago`, `a few weeks ago`: N units back, with
///   slack either side (a day per week of distance, 4 days, 15 days, half a
///   year);
/// - `last|past week/month/year`: two units back to `asked`;
///   `this week/month/year`: the start of the period to `asked`;
/// - `yesterday`, `today`, `tonight`, `this morning`, `last weekend`,
///   `last <weekday>`;
/// - `last|this <season>`;
/// - `<month> <day>[, <year>]`, `[in|on|during|since] <month> [<year>]`
///   ("may" only after a preposition or before a number), `in <year>`.
pub fn question_window(question: &str, asked: NaiveDate) -> Option<TimeWindow> {
    let lower = question.to_lowercase();
    let words: Vec<String> = lower
        .split(|c: char| c.is_whitespace() || c == ',' || c == '/' || c == '-')
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .trim_end_matches("'s")
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let w: Vec<&str> = words.iter().map(String::as_str).collect();

    // last/past N units (to now).
    for i in 0..w.len() {
        if !matches!(w[i], "last" | "past" | "previous") {
            continue;
        }
        if let Some((n, at)) = count_at(&w, i + 1) {
            let mut j = at + 1;
            if matches!(w.get(j), Some(&"few" | &"couple")) {
                j += 1;
            }
            if w.get(j) == Some(&"of") {
                j += 1;
            }
            if let Some(unit) = w.get(j).and_then(|x| unit_of(x)) {
                let from = back(asked, n, unit)?;
                return window(from, asked, &w[i..=j]);
            }
        }
    }
    // N units ago.
    // "how many days ago" asks for a distance; it names no time.
    if let Some(ago) = w.iter().position(|x| *x == "ago")
        && ago >= 1
        && let Some(unit) = unit_of(w[ago - 1])
        && !(ago >= 2 && matches!(w[ago - 2], "many" | "much"))
    {
        let mut start = ago - 1;
        let n = match ago.checked_sub(2) {
            Some(k) => match count_at(&w, k) {
                Some((n, _)) => {
                    start = k;
                    if matches!(w[k], "few" | "couple") && k > 0 && matches!(w[k - 1], "a" | "an") {
                        start = k - 1;
                    }
                    n
                }
                None => 1,
            },
            None => 1,
        };
        let center = back(asked, n, unit)?;
        let s = slack(n, unit);
        return window(center - s, (center + s).min(asked), &w[start..=ago]);
    }
    for i in 0..w.len() {
        let next = w.get(i + 1).copied();
        match (w[i], next) {
            ("last" | "past" | "previous", Some(u)) if unit_of(u).is_some() => {
                let unit = unit_of(u)?;
                let from = back(asked, 2, unit)?;
                return window(from, asked, &w[i..=i + 1]);
            }
            ("this", Some("week")) => {
                let from =
                    asked - Duration::days(i64::from(asked.weekday().num_days_from_monday()));
                return window(from, asked, &w[i..=i + 1]);
            }
            ("this", Some("month")) => {
                return window(asked.with_day(1)?, asked, &w[i..=i + 1]);
            }
            ("this", Some("year")) => {
                return window(asked.with_ordinal(1)?, asked, &w[i..=i + 1]);
            }
            ("yesterday", _) => {
                let d = asked.pred_opt()?;
                return window(d, d, &w[i..=i]);
            }
            ("today" | "tonight", _) | ("this", Some("morning" | "afternoon" | "evening")) => {
                return window(asked, asked, &w[i..=i]);
            }
            ("last" | "past", Some("weekend")) => {
                let sat = previous(asked, Weekday::Sat);
                return window(sat, (sat + Duration::days(1)).min(asked), &w[i..=i + 1]);
            }
            ("last" | "past", Some(day)) if weekday_of(day).is_some() => {
                let d = previous(asked, weekday_of(day)?);
                return window(d, d, &w[i..=i + 1]);
            }
            ("last" | "this", Some(season)) => {
                if let Some(&(_, a, b)) = SEASONS.iter().find(|s| s.0 == season) {
                    // The most recent such season that has started.
                    let mut y = asked.year();
                    let starts = |y: i32| NaiveDate::from_ymd_opt(y, a, 1);
                    while starts(y)? > asked {
                        y -= 1;
                    }
                    if w[i] == "last" && month_end(if b < a { y + 1 } else { y }, b)? >= asked {
                        y -= 1;
                    }
                    let end = month_end(if b < a { y + 1 } else { y }, b)?.min(asked);
                    return window(starts(y)?, end, &w[i..=i + 1]);
                }
            }
            _ => {}
        }
    }
    // Calendar months and years.
    for i in 0..w.len() {
        let Some(month) = month_of(w[i]) else {
            continue;
        };
        // Month words must be whole month names or common abbreviations.
        if w[i].len() < 3
            || ![
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct",
                "nov", "dec",
            ]
            .iter()
            .any(|a| w[i].starts_with(a))
        {
            continue;
        }
        let prep = i > 0
            && matches!(
                w[i - 1],
                "in" | "on" | "during" | "since" | "of" | "early" | "late" | "mid"
            );
        let day = w.get(i + 1).and_then(|x| crate::session_extract::day_of(x));
        let year = w
            .get(i + 1 + usize::from(day.is_some()))
            .and_then(|x| year_of(x));
        if w[i] == "may" && !prep && day.is_none() && year.is_none() {
            continue;
        }
        if w[i] == "march" && !prep && day.is_none() && year.is_none() {
            continue;
        }
        let end_at = i + usize::from(day.is_some()) + usize::from(year.is_some());
        let y = match year {
            Some(y) => y,
            None => {
                // The latest such month that has started by `asked`.
                if NaiveDate::from_ymd_opt(asked.year(), month, 1)? <= asked {
                    asked.year()
                } else {
                    asked.year() - 1
                }
            }
        };
        return match day {
            Some(d) => {
                let d = NaiveDate::from_ymd_opt(y, month, d)?;
                window(d, d, &w[i..=end_at])
            }
            None => window(
                NaiveDate::from_ymd_opt(y, month, 1)?,
                month_end(y, month)?.min(asked),
                &w[i..=end_at],
            ),
        };
    }
    for i in 1..w.len() {
        if matches!(w[i - 1], "in" | "during")
            && let Some(y) = year_of(w[i])
            && y <= asked.year()
        {
            return window(
                NaiveDate::from_ymd_opt(y, 1, 1)?,
                NaiveDate::from_ymd_opt(y, 12, 31)?.min(asked),
                &w[i - 1..=i],
            );
        }
    }
    None
}

/// Whole days from `d` to `asked`; negative when `d` is later.
pub fn days_before(d: NaiveDate, asked: NaiveDate) -> i64 {
    (asked - d).num_days()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn win(q: &str, asked: NaiveDate) -> Option<(NaiveDate, NaiveDate, String)> {
        question_window(q, asked).map(|w| (w.from, w.to, w.phrase))
    }

    #[test]
    fn reads_time_references_from_questions() {
        // 2023-05-24 is a Wednesday.
        let a = d(2023, 5, 24);
        let cases = [
            (
                "How many MCU films did I watch in the last 3 months?",
                d(2023, 2, 24),
                a,
                "last 3 months",
            ),
            (
                "What is the total weight of the feed I bought in the past two months?",
                d(2023, 3, 24),
                a,
                "past two months",
            ),
            (
                "What did I buy over the past few weeks?",
                d(2023, 5, 3),
                a,
                "past few weeks",
            ),
            (
                "Which pair of shoes did I clean last month?",
                d(2023, 3, 24),
                a,
                "last month",
            ),
            (
                "How much better was my mileage a few months ago?",
                d(2023, 2, 9),
                d(2023, 3, 11),
                "a few months ago",
            ),
            (
                "Who did I meet two weeks ago?",
                d(2023, 5, 6),
                d(2023, 5, 14),
                "two weeks ago",
            ),
            (
                "What did I cook yesterday?",
                d(2023, 5, 23),
                d(2023, 5, 23),
                "yesterday",
            ),
            (
                "Where was I last Saturday?",
                d(2023, 5, 20),
                d(2023, 5, 20),
                "last saturday",
            ),
            (
                "What did I do last weekend?",
                d(2023, 5, 20),
                d(2023, 5, 21),
                "last weekend",
            ),
            (
                "What did I read this month?",
                d(2023, 5, 1),
                a,
                "this month",
            ),
            (
                "Which concerts did I go to in March?",
                d(2023, 3, 1),
                d(2023, 3, 31),
                "march",
            ),
            (
                "What happened on May 3rd?",
                d(2023, 5, 3),
                d(2023, 5, 3),
                "may 3rd",
            ),
            (
                "Where did I travel in December?",
                d(2022, 12, 1),
                d(2022, 12, 31),
                "december",
            ),
            (
                "What car did I drive in 2021?",
                d(2021, 1, 1),
                d(2021, 12, 31),
                "in 2021",
            ),
            (
                "Where did I go last summer?",
                d(2022, 6, 1),
                d(2022, 8, 31),
                "last summer",
            ),
        ];
        for (q, from, to, phrase) in cases {
            assert_eq!(win(q, a), Some((from, to, phrase.to_string())), "{q}");
        }
        for q in [
            "How many days ago did I meet Emma?",
            "Which event happened first, the meeting with Rachel or the parade?",
            "What is my current highest score in Ticket to Ride?",
            "I may want to repaint; any tips?",
            "How did the march go?",
        ] {
            assert_eq!(win(q, a), None, "{q}");
        }
    }

    #[test]
    fn counts_days_before_the_question() {
        assert_eq!(days_before(d(2023, 5, 1), d(2023, 5, 24)), 23);
        assert_eq!(days_before(d(2023, 5, 25), d(2023, 5, 24)), -1);
        assert!(
            question_window("last week", d(2023, 5, 24))
                .unwrap()
                .contains(d(2023, 5, 15))
        );
    }
}
