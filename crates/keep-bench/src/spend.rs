//! A persistent spend ledger shared by every USD-metered run.
//!
//! Each run's own [`keep_providers::BudgetLedger`] stops it at its cost cap:
//! the manifest's `max_cost_usd`, or `--max-run-usd` when that is lower. This
//! ledger bounds the total across runs: before any call, a run reserves its
//! full cost cap here, and is refused if settled spend plus every open
//! reservation plus this one would cross the cap. When
//! the run ends, the reservation is settled to what its budget ledger
//! actually charged.
//!
//! The file is append-only JSON lines, so it doubles as a per-run spend
//! report. A run that dies without settling (a killed process) leaves its
//! full reservation open, which keeps the cap safe; `keep-bench spend settle`
//! closes it by hand once the real charge is known.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use keep_providers::BudgetLedger;
use serde::{Deserialize, Serialize};

pub const DEFAULT_CAP_USD: f64 = 300.0;
pub const DEFAULT_PATH: &str = "results/spend-ledger.jsonl";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Entry {
    Reserve {
        id: String,
        at: String,
        kind: String,
        manifest_id: String,
        manifest_hash: String,
        reserved_usd: f64,
    },
    Settle {
        id: String,
        at: String,
        spent_usd: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSpend {
    pub id: String,
    pub kind: String,
    pub manifest_id: String,
    pub reserved_usd: f64,
    /// `None` while the reservation is open.
    pub spent_usd: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpendStatus {
    pub cap_usd: f64,
    pub settled_usd: f64,
    pub open_reserved_usd: f64,
    pub runs: Vec<RunSpend>,
}

impl SpendStatus {
    /// Settled spend plus every open reservation.
    pub fn committed_usd(&self) -> f64 {
        self.settled_usd + self.open_reserved_usd
    }
}

#[derive(Clone, Debug)]
pub struct SpendLedger {
    path: PathBuf,
    cap_usd: f64,
    run_cap_usd: Option<f64>,
}

/// Holds the ledger's lock file for one read-modify-append.
struct Lock(PathBuf);

impl Lock {
    fn acquire(path: &Path) -> anyhow::Result<Self> {
        let lock = path.with_extension("lock");
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let started = Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
            {
                Ok(_) => return Ok(Self(lock)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if started.elapsed() > Duration::from_secs(30) {
                        anyhow::bail!(
                            "spend ledger lock {} has been held for 30s; if no keep-bench process is running, delete it",
                            lock.display()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

impl SpendLedger {
    pub fn new(path: impl Into<PathBuf>, cap_usd: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cap_usd.is_finite() && cap_usd >= 0.0,
            "spend cap must be a non-negative number of dollars"
        );
        Ok(Self {
            path: path.into(),
            cap_usd,
            run_cap_usd: None,
        })
    }

    /// Lower every run's cost cap to `usd` when it is below the manifest's
    /// `max_cost_usd`. It never raises a manifest's cap.
    pub fn with_run_cap(mut self, usd: Option<f64>) -> anyhow::Result<Self> {
        if let Some(usd) = usd {
            anyhow::ensure!(
                usd.is_finite() && usd > 0.0,
                "per-run cap must be a positive number of dollars"
            );
        }
        self.run_cap_usd = usd;
        Ok(self)
    }

    /// The cost cap a run with this manifest cap is held to.
    pub fn run_cap(&self, manifest_max_cost_usd: f64) -> f64 {
        self.run_cap_usd
            .map_or(manifest_max_cost_usd, |c| c.min(manifest_max_cost_usd))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read(&self) -> anyhow::Result<Vec<Entry>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        text.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(i, l)| {
                serde_json::from_str(l)
                    .map_err(|e| anyhow::anyhow!("{} line {}: {e}", self.path.display(), i + 1))
            })
            .collect()
    }

    fn append(&self, entry: &Entry) -> anyhow::Result<()> {
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{}", serde_json::to_string(entry)?)?;
        f.sync_all()?;
        Ok(())
    }

    fn status_of(&self, entries: &[Entry]) -> anyhow::Result<SpendStatus> {
        let mut runs: Vec<RunSpend> = Vec::new();
        for e in entries {
            match e {
                Entry::Reserve {
                    id,
                    kind,
                    manifest_id,
                    reserved_usd,
                    ..
                } => runs.push(RunSpend {
                    id: id.clone(),
                    kind: kind.clone(),
                    manifest_id: manifest_id.clone(),
                    reserved_usd: *reserved_usd,
                    spent_usd: None,
                }),
                Entry::Settle { id, spent_usd, .. } => {
                    let run = runs.iter_mut().find(|r| &r.id == id).ok_or_else(|| {
                        anyhow::anyhow!("spend ledger settles unknown reservation {id}")
                    })?;
                    run.spent_usd = Some(*spent_usd);
                }
            }
        }
        Ok(SpendStatus {
            cap_usd: self.cap_usd,
            // An empty f64 sum is -0.0; adding 0.0 keeps reports at "$0.00".
            settled_usd: runs.iter().filter_map(|r| r.spent_usd).sum::<f64>() + 0.0,
            open_reserved_usd: runs
                .iter()
                .filter(|r| r.spent_usd.is_none())
                .map(|r| r.reserved_usd)
                .sum::<f64>()
                + 0.0,
            runs,
        })
    }

    pub fn status(&self) -> anyhow::Result<SpendStatus> {
        self.status_of(&self.read()?)
    }

    /// Reserve `amount_usd` for a run, or refuse if the cap would be crossed.
    pub fn reserve(
        &self,
        id: &str,
        kind: &str,
        manifest_id: &str,
        manifest_hash: &str,
        amount_usd: f64,
    ) -> anyhow::Result<()> {
        let _lock = Lock::acquire(&self.path)?;
        let status = self.status_of(&self.read()?)?;
        anyhow::ensure!(
            !status.runs.iter().any(|r| r.id == id),
            "spend ledger already has a reservation {id}"
        );
        let committed = status.committed_usd();
        if committed + amount_usd > self.cap_usd {
            anyhow::bail!(
                "spend cap refused this run: ${:.2} settled + ${:.2} reserved by open runs + ${:.2} for this run = ${:.2}, over the ${:.2} cap in {}; \
                 lower budget.max_cost_usd, raise --spend-cap-usd, or settle stale reservations with `keep-bench spend settle`",
                status.settled_usd,
                status.open_reserved_usd,
                amount_usd,
                committed + amount_usd,
                self.cap_usd,
                self.path.display()
            );
        }
        self.append(&Entry::Reserve {
            id: id.into(),
            at: now(),
            kind: kind.into(),
            manifest_id: manifest_id.into(),
            manifest_hash: manifest_hash.into(),
            reserved_usd: amount_usd,
        })
    }

    /// Close an open reservation at the actual spend.
    pub fn settle(&self, id: &str, spent_usd: f64, note: Option<String>) -> anyhow::Result<()> {
        anyhow::ensure!(
            spent_usd.is_finite() && spent_usd >= 0.0,
            "spent amount must be a non-negative number of dollars"
        );
        let _lock = Lock::acquire(&self.path)?;
        let status = self.status_of(&self.read()?)?;
        match status.runs.iter().find(|r| r.id == id) {
            None => anyhow::bail!("no reservation {id} in {}", self.path.display()),
            Some(r) if r.spent_usd.is_some() => {
                anyhow::bail!("reservation {id} is already settled")
            }
            Some(_) => {}
        }
        self.append(&Entry::Settle {
            id: id.into(),
            at: now(),
            spent_usd,
            note,
        })
    }

    /// Reserve, and settle automatically from `budget` when the guard drops,
    /// on success, error or panic alike.
    pub fn guard(
        &self,
        id: &str,
        kind: &str,
        manifest_id: &str,
        manifest_hash: &str,
        amount_usd: f64,
        budget: Arc<BudgetLedger>,
    ) -> anyhow::Result<SpendGuard> {
        self.reserve(id, kind, manifest_id, manifest_hash, amount_usd)?;
        Ok(SpendGuard {
            ledger: self.clone(),
            id: id.into(),
            budget,
        })
    }
}

/// Settles its reservation to the budget ledger's charged cost on drop. The
/// budget ledger counts in-flight and failed calls at their worst case, so
/// the settled figure never under-reports spend.
pub struct SpendGuard {
    ledger: SpendLedger,
    id: String,
    budget: Arc<BudgetLedger>,
}

impl Drop for SpendGuard {
    fn drop(&mut self) {
        let spent = self.budget.usage().cost_usd;
        if let Err(e) = self.ledger.settle(&self.id, spent, None) {
            eprintln!(
                "warning: could not settle spend reservation {} at ${spent:.4}: {e:#}",
                self.id
            );
        }
    }
}

pub fn markdown(s: &SpendStatus) -> String {
    let mut out = format!(
        "Spend: ${:.4} settled, ${:.2} reserved by open runs, ${:.2} committed of the ${:.2} cap (${:.2} left).\n\n",
        s.settled_usd,
        s.open_reserved_usd,
        s.committed_usd(),
        s.cap_usd,
        (s.cap_usd - s.committed_usd()).max(0.0)
    );
    if !s.runs.is_empty() {
        out.push_str(
            "| Run | Kind | Manifest | Reserved USD | Spent USD |\n|---|---|---|---|---|\n",
        );
        for r in &s.runs {
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {} |\n",
                r.id,
                r.kind,
                r.manifest_id,
                r.reserved_usd,
                r.spent_usd
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "open".into())
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use keep_providers::{BudgetCaps, Price};

    fn budget() -> Arc<BudgetLedger> {
        Arc::new(BudgetLedger::new(BudgetCaps {
            max_calls: 10,
            max_input_tokens: 1_000_000,
            max_output_tokens: 1_000_000,
            max_cost_usd: 100.0,
        }))
    }

    #[test]
    fn reservations_count_against_the_cap_until_settled() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = SpendLedger::new(dir.path().join("r/spend.jsonl"), 100.0).unwrap();
        ledger.reserve("a", "run", "m", "h", 60.0).unwrap();
        let err = ledger.reserve("b", "run", "m", "h", 50.0).unwrap_err();
        assert!(err.to_string().contains("spend cap refused"), "{err}");
        ledger.settle("a", 12.5, None).unwrap();
        ledger.reserve("b", "run", "m", "h", 50.0).unwrap();
        let s = ledger.status().unwrap();
        assert_eq!((s.settled_usd, s.open_reserved_usd), (12.5, 50.0));
        // Exactly reaching the cap is allowed; one cent over is not.
        ledger.reserve("c", "run", "m", "h", 37.5).unwrap();
        assert!(ledger.reserve("d", "run", "m", "h", 0.01).is_err());
        assert!(ledger.settle("a", 1.0, None).is_err(), "settles once");
        assert!(ledger.settle("zzz", 1.0, None).is_err());
        assert!(
            ledger.reserve("b", "run", "m", "h", 0.0).is_err(),
            "ids are unique"
        );
    }

    #[test]
    fn guard_settles_to_charged_cost_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = SpendLedger::new(dir.path().join("spend.jsonl"), 10.0).unwrap();
        let b = budget();
        let price = Price {
            input_per_mtok_usd: 1_000_000.0,
            output_per_mtok_usd: 0.0,
        };
        {
            let _g = ledger
                .guard("run-1", "run", "m", "h", 8.0, b.clone())
                .unwrap();
            let r = b.reserve(5, 0, &price).unwrap();
            b.settle(r, 3, 0, &price);
            assert!(ledger.reserve("x", "run", "m", "h", 2.5).is_err());
        }
        let s = ledger.status().unwrap();
        assert_eq!((s.settled_usd, s.open_reserved_usd), (3.0, 0.0));
        ledger.reserve("x", "run", "m", "h", 7.0).unwrap();
        assert!(!dir.path().join("spend.lock").exists());
    }
}
