//! `keep-bench`: the command-line entry point for Keep's research harness.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use keep_bench::dataset::DatasetKind;
use keep_bench::manifest::{LoadedManifest, StorageProfile, has_errors};
use keep_bench::runner::{Mode, RunOptions};
use keep_bench::{ablation, calibration, compare, dryrun, fetch, prepare, report, runner};

#[derive(Parser)]
#[command(
    name = "keep-bench",
    version,
    about = "Preregistered benchmark harness for Keep"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum FetchDataset {
    Locomo,
    LongmemevalS,
    LongmemevalOracle,
    LongmemevalM,
}

impl From<FetchDataset> for DatasetKind {
    fn from(d: FetchDataset) -> Self {
        match d {
            FetchDataset::Locomo => DatasetKind::Locomo,
            FetchDataset::LongmemevalS => DatasetKind::LongmemevalS,
            FetchDataset::LongmemevalOracle => DatasetKind::LongmemevalOracle,
            FetchDataset::LongmemevalM => DatasetKind::LongmemevalM,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Profile {
    InMemory,
    PostgresVector,
    PostgresQdrant,
}

impl From<Profile> for StorageProfile {
    fn from(p: Profile) -> Self {
        match p {
            Profile::InMemory => StorageProfile::InMemory,
            Profile::PostgresVector => StorageProfile::PostgresVector,
            Profile::PostgresQdrant => StorageProfile::PostgresQdrant,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Download public datasets into the data directory and verify checksums.
    Fetch {
        /// Datasets to fetch; defaults to LoCoMo, LongMemEval-S and the oracle split.
        #[arg(long = "dataset", value_enum)]
        datasets: Vec<FetchDataset>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Re-download files that already exist.
        #[arg(long)]
        force: bool,
    },
    /// Check a manifest and every file it pins. Exits non-zero on errors.
    ValidateManifest {
        manifest: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print the holdout commitment to paste into `holdout.commitment`.
    Seal { manifest: PathBuf },
    /// Compute the length-only full-context fit mask over the whole dataset.
    FitMask {
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Build every context and estimate tokens, calls and cost without
    /// calling any model or database.
    DryRun {
        manifest: PathBuf,
        #[arg(long)]
        json: bool,
        /// Estimate a confirmation manifest on the sealed holdout it would
        /// score. Counts lengths only; calls and scores nothing.
        #[arg(long)]
        unseal: bool,
    },
    /// Build every context with the manifest's local embedder, reranker and
    /// storage, and report evidence recall and context tokens per arm. Calls
    /// no answer, judge or remote model; development questions only.
    Contexts {
        manifest: PathBuf,
        /// Comma-separated arm names; defaults to every arm.
        #[arg(long, value_delimiter = ',')]
        arms: Vec<String>,
        /// File of development question ids, one per line, to restrict to.
        #[arg(long)]
        questions: Option<PathBuf>,
        /// Parent directory for the run directory (kept out of `results`).
        #[arg(long, default_value = "target/keep-contexts")]
        out: PathBuf,
        /// Print per-phase context-building timings to stderr.
        #[arg(long)]
        timings: bool,
        /// Allow a protocol with `split = "full"` to read the sealed
        /// holdout. Spends the holdout; needs explicit approval.
        #[arg(long)]
        consume_holdout: bool,
    },
    /// Write each arm's retrieval in the per-question format of Mem0's
    /// open memory-benchmarks harness, for its `--evaluate-only` mode. No
    /// answer or judge model is called. See docs/benchmarks/parity.md.
    Export {
        manifest: PathBuf,
        /// Comma-separated arm names; defaults to every arm.
        #[arg(long, value_delimiter = ',')]
        arms: Vec<String>,
        /// File of development question ids, one per line, to restrict to.
        #[arg(long)]
        questions: Option<PathBuf>,
        /// New, empty output directory.
        #[arg(long)]
        out: PathBuf,
        /// Harness project name (`predicted_<project>` directories).
        #[arg(long, default_value = "keep")]
        project: String,
        /// Export each context as one memory instead of one per line.
        #[arg(long)]
        single_memory: bool,
        /// Allow a protocol with `split = "full"` to read the sealed
        /// holdout. Spends the holdout; needs explicit approval.
        #[arg(long)]
        consume_holdout: bool,
    },
    /// Write one arm's answers from a finished run as the hypothesis file
    /// (`{"question_id", "hypothesis"}` per line) read by the official
    /// LongMemEval evaluation script.
    Hypotheses {
        run_dir: PathBuf,
        #[arg(long)]
        arm: String,
        /// Seed to export; defaults to the arm's first seed.
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Extract facts from every session of the development histories (or
    /// of the selected questions' histories) into the on-disk cache that the
    /// k3 and mk3 arms read. Cached sessions are not called again.
    Extract {
        manifest: PathBuf,
        #[arg(long)]
        allow_model_calls: bool,
        /// File of development question ids, one per line, to restrict to.
        #[arg(long)]
        questions: Option<PathBuf>,
        #[command(flatten)]
        spend: SpendArgs,
    },
    /// Run a manifest. Remote models are refused unless both the manifest
    /// and this flag allow them.
    Run {
        manifest: PathBuf,
        #[arg(long)]
        allow_model_calls: bool,
        /// Run on the sealed holdout (confirmation phase only).
        #[arg(long)]
        unseal: bool,
        /// Allow a protocol with `split = "full"` to score the sealed
        /// holdout, as published numbers do. Spends the holdout; needs
        /// explicit approval.
        #[arg(long)]
        consume_holdout: bool,
        /// Comma-separated arm names; defaults to every arm.
        #[arg(long, value_delimiter = ',')]
        arms: Vec<String>,
        /// File of question ids, one per line, to restrict to (smoke tests).
        #[arg(long)]
        questions: Option<PathBuf>,
        #[arg(long, default_value = "results")]
        results: PathBuf,
        /// Print per-phase context-building timings to stderr (also
        /// `KEEP_TIMINGS=1`). Names, counts and durations only.
        #[arg(long)]
        timings: bool,
        #[command(flatten)]
        spend: SpendArgs,
    },
    /// Paired comparison of a candidate arm against comparators, with the
    /// preregistered superiority criteria.
    Compare {
        /// Run directories (one or more per storage profile).
        #[arg(long = "run", required = true)]
        runs: Vec<PathBuf>,
        #[arg(long)]
        candidate: String,
        #[arg(long = "against", required = true)]
        against: Vec<String>,
        /// Storage profiles that must each pass.
        #[arg(long = "require-profile", value_enum)]
        require_profiles: Vec<Profile>,
        #[arg(long)]
        json: bool,
        /// Admit non-canonical runs (Copilot CLI or other development-track
        /// runs) for an internal comparison. All runs must be non-canonical
        /// and share answer and judge models; every output is stamped
        /// INTERNAL / NON-CANONICAL and the gate never passes.
        #[arg(long)]
        allow_non_canonical: bool,
        /// Also write comparison.md and comparison.json to this new directory.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Regenerate summary.md and summary.json for a run directory.
    Report { run: PathBuf },
    /// Measure the judge against probes and optional human labels.
    CalibrateJudge {
        manifest: PathBuf,
        /// JSONL of labelled items.
        #[arg(long)]
        labels: Option<PathBuf>,
        /// Probes per type built from development questions; 0 disables them.
        #[arg(long, default_value_t = 25)]
        probes_per_type: usize,
        #[arg(long)]
        allow_model_calls: bool,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        spend: SpendArgs,
    },
    /// Print the Keep ablation arms as manifest TOML.
    AblationGrid,
    /// Show or correct the cross-run spend ledger.
    Spend {
        #[command(subcommand)]
        action: SpendAction,
        #[command(flatten)]
        spend: SpendArgs,
    },
}

#[derive(clap::Args)]
struct SpendArgs {
    /// Append-only ledger of USD spend across runs.
    #[arg(long, default_value = keep_bench::spend::DEFAULT_PATH)]
    spend_ledger: PathBuf,
    /// Total USD that all runs sharing the ledger may commit.
    #[arg(long, default_value_t = keep_bench::spend::DEFAULT_CAP_USD)]
    spend_cap_usd: f64,
    /// Hold each run to this many USD when it is below the manifest's
    /// `budget.max_cost_usd` (never raises it). For smoke runs on a few
    /// questions.
    #[arg(long)]
    max_run_usd: Option<f64>,
}

impl SpendArgs {
    fn ledger(&self) -> anyhow::Result<keep_bench::spend::SpendLedger> {
        keep_bench::spend::SpendLedger::new(&self.spend_ledger, self.spend_cap_usd)?
            .with_run_cap(self.max_run_usd)
    }
}

#[derive(Subcommand)]
enum SpendAction {
    /// Print settled spend, open reservations and what is left.
    Show {
        #[arg(long)]
        json: bool,
    },
    /// Close a reservation left open by a run that was killed, at the
    /// spend the provider actually billed for it.
    Settle {
        id: String,
        #[arg(long)]
        usd: f64,
        #[arg(long)]
        note: Option<String>,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match dispatch(Cli::parse().command).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Ids from a file, one per line; blank lines and `#` comments skipped.
fn read_ids(path: Option<&Path>) -> anyhow::Result<Option<std::collections::BTreeSet<String>>> {
    let Some(path) = path else { return Ok(None) };
    Ok(Some(
        std::fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(String::from)
            .collect(),
    ))
}

fn print_json<T: serde::Serialize>(v: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

async fn dispatch(command: Command) -> anyhow::Result<ExitCode> {
    match command {
        Command::Fetch {
            datasets,
            data_dir,
            force,
        } => {
            let kinds: Vec<DatasetKind> = datasets.into_iter().map(Into::into).collect();
            for src in fetch::selection(&kinds) {
                eprintln!(
                    "fetching {} (~{} MB, {})",
                    src.url,
                    src.approx_bytes / 1_000_000,
                    src.license
                );
                let f = fetch::fetch(src, &data_dir, force).await?;
                println!(
                    "{} {} sha256 {} {}",
                    if f.skipped { "present" } else { "fetched" },
                    f.path.display(),
                    f.sha256,
                    if f.verified {
                        "(verified)"
                    } else {
                        "(no pinned checksum: record this hash)"
                    }
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::ValidateManifest { manifest, json } => {
            let lm = LoadedManifest::load(&manifest)?;
            let issues = prepare::check(&lm);
            let problems = lm.manifest.dispatch_problems();
            let failed = has_errors(&issues);
            if json {
                print_json(&serde_json::json!({
                    "manifest_hash": lm.manifest.hash(),
                    "valid": !failed,
                    "issues": issues,
                    "dispatch_problems": problems,
                }))?;
            } else {
                println!("manifest {} ({})", lm.manifest.id, lm.path.display());
                println!("hash {}", lm.manifest.hash());
                for i in &issues {
                    println!(
                        "{} {}: {}",
                        if i.error { "error" } else { "warning" },
                        i.field,
                        i.message
                    );
                }
                if lm.manifest.uses_remote_models() {
                    for p in &problems {
                        println!("dispatch blocked: {p}");
                    }
                }
                println!("{}", if failed { "invalid" } else { "valid" });
            }
            Ok(if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            })
        }
        Command::Seal { manifest } => {
            let lm = LoadedManifest::load(&manifest)?;
            let (commitment, held, total) = prepare::seal(&lm)?;
            println!("commitment = \"{commitment}\"");
            eprintln!("{held} of {total} histories are in the holdout; their ids are not printed");
            Ok(ExitCode::SUCCESS)
        }
        Command::FitMask { manifest, out } => {
            let lm = LoadedManifest::load(&manifest)?;
            let mask = prepare::compute_fit_mask(&lm)?;
            write_new(&out, &serde_json::to_vec_pretty(&mask)?)?;
            let sha = fetch::file_sha256(&out)?;
            println!(
                "fit fraction {:.1}% over {} questions (tokenizer {}{}); gate-eligible: {}",
                mask.fit_fraction * 100.0,
                mask.entries.len(),
                mask.tokenizer,
                if mask.tokenizer_exact {
                    ""
                } else {
                    ", approximate"
                },
                mask.gate_eligible
            );
            println!(
                "pin it with: fit_mask = {{ path = \"{}\", sha256 = \"{sha}\" }}",
                out.display()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::DryRun {
            manifest,
            json,
            unseal,
        } => {
            let lm = LoadedManifest::load(&manifest)?;
            let r = dryrun::dry_run_with(&lm, unseal).await?;
            if json {
                print_json(&r)?;
            } else {
                print!("{}", dryrun::markdown(&r));
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Extract {
            manifest,
            allow_model_calls,
            questions,
            spend,
        } => {
            let lm = LoadedManifest::load(&manifest)?;
            let questions = read_ids(questions.as_deref())?;
            let stats = runner::extract(
                &lm,
                &RunOptions {
                    mode: Mode::Execute,
                    unseal: false,
                    allow_model_calls,
                    results_root: None,
                    spend: Some(spend.ledger()?),
                },
                &runner::Selection {
                    arms: None,
                    questions,
                    ..Default::default()
                },
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
            Ok(if stats.failed() == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Command::Contexts {
            manifest,
            arms,
            questions,
            out,
            timings,
            consume_holdout,
        } => {
            if timings {
                keep_providers::timing::set_enabled(true);
            }
            let lm = LoadedManifest::load(&manifest)?;
            let selection = runner::Selection {
                arms: (!arms.is_empty()).then(|| arms.into_iter().collect()),
                questions: read_ids(questions.as_deref())?,
                consume_holdout,
                capture_context_text: false,
            };
            let run = runner::run_selected(
                &lm,
                &RunOptions {
                    mode: Mode::Contexts,
                    unseal: false,
                    allow_model_calls: false,
                    results_root: Some(out),
                    spend: None,
                },
                &selection,
            )
            .await?;
            let summary = report::summarize(&run.meta, &lm.manifest, &run.records);
            print!("{}", report::markdown(&summary));
            if let Some(dir) = run.dir {
                println!("\nartifacts: {}", dir.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Export {
            manifest,
            arms,
            questions,
            out,
            project,
            single_memory,
            consume_holdout,
        } => {
            let lm = LoadedManifest::load(&manifest)?;
            let report = keep_bench::export::export(
                &lm,
                keep_bench::export::ExportRequest {
                    out,
                    project,
                    single_memory,
                    selection: runner::Selection {
                        arms: (!arms.is_empty()).then(|| arms.into_iter().collect()),
                        questions: read_ids(questions.as_deref())?,
                        consume_holdout,
                        capture_context_text: true,
                    },
                },
            )
            .await?;
            for a in &report.arms {
                println!(
                    "{}: {} files in {} ({} missing)",
                    a.arm,
                    a.written,
                    a.output_dir,
                    a.missing.len()
                );
            }
            for c in &report.commands {
                println!("\n{c}");
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Hypotheses {
            run_dir,
            arm,
            seed,
            out,
        } => {
            let run = keep_bench::compare::load_run(&run_dir)?;
            anyhow::ensure!(
                run.manifest
                    .dataset
                    .kind
                    .as_str()
                    .starts_with("longmemeval"),
                "hypothesis files are for LongMemEval runs"
            );
            let (hyps, unanswered) = keep_bench::export::hypotheses(&run.records, &arm, seed)?;
            keep_bench::export::write_hypotheses(&out, &hyps)?;
            println!(
                "{} hypotheses ({unanswered} unanswered, written empty) -> {}",
                hyps.len(),
                out.display()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Run {
            manifest,
            allow_model_calls,
            unseal,
            consume_holdout,
            arms,
            questions,
            results,
            timings,
            spend,
        } => {
            if timings {
                keep_providers::timing::set_enabled(true);
            }
            let lm = LoadedManifest::load(&manifest)?;
            let selection = runner::Selection {
                arms: (!arms.is_empty()).then(|| arms.into_iter().collect()),
                questions: read_ids(questions.as_deref())?,
                consume_holdout,
                capture_context_text: false,
            };
            let out = runner::run_selected(
                &lm,
                &RunOptions {
                    mode: Mode::Execute,
                    unseal,
                    allow_model_calls,
                    results_root: Some(results),
                    spend: Some(spend.ledger()?),
                },
                &selection,
            )
            .await?;
            let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
            print!("{}", report::markdown(&summary));
            if let Some(dir) = out.dir {
                println!("\nartifacts: {}", dir.display());
            }
            if lm.manifest.uses_openai_compatible() {
                println!(
                    "\n{}",
                    keep_bench::spend::markdown(&spend.ledger()?.status()?)
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Compare {
            runs,
            candidate,
            against,
            require_profiles,
            json,
            allow_non_canonical,
            out,
        } => {
            let loaded = runs
                .iter()
                .map(|d| compare::load_run(d))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let profiles: Vec<StorageProfile> =
                require_profiles.into_iter().map(Into::into).collect();
            let g = compare::evaluate_with(
                &loaded,
                &candidate,
                &against,
                &profiles,
                compare::CompareOptions {
                    allow_non_canonical,
                },
            )?;
            let md = compare::markdown(&g);
            if let Some(dir) = out {
                anyhow::ensure!(
                    !dir.exists(),
                    "{} already exists; comparison reports are never overwritten",
                    dir.display()
                );
                std::fs::create_dir_all(&dir)?;
                keep_bench::artifacts::write_json(&dir.join("comparison.json"), &g)?;
                std::fs::write(dir.join("comparison.md"), &md)?;
                eprintln!("wrote {}", dir.display());
            }
            if json {
                print_json(&g)?;
            } else {
                print!("{md}");
            }
            Ok(if g.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            })
        }
        Command::Report { run } => {
            let r = compare::load_run(&run)?;
            let summary = report::summarize(&r.meta, &r.manifest, &r.records);
            keep_bench::artifacts::write_json(&run.join("summary.json"), &summary)?;
            let md = report::markdown(&summary);
            std::fs::write(run.join("summary.md"), &md)?;
            print!("{md}");
            Ok(ExitCode::SUCCESS)
        }
        Command::CalibrateJudge {
            manifest,
            labels,
            probes_per_type,
            allow_model_calls,
            json,
            spend,
        } => {
            let lm = LoadedManifest::load(&manifest)?;
            let spend = spend.ledger()?;
            let (judge, budget) =
                runner::judge_model(&lm, allow_model_calls, Some(spend.clone())).await?;
            let _spend = match budget {
                Some(budget) if lm.manifest.uses_openai_compatible() => Some(spend.guard(
                    &format!(
                        "calibrate-{}-{}",
                        chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ"),
                        lm.hash.short()
                    ),
                    "calibrate-judge",
                    &lm.manifest.id,
                    &lm.hash.to_string(),
                    budget.caps().max_cost_usd,
                    budget,
                )?),
                _ => None,
            };
            let mut items = Vec::new();
            if probes_per_type > 0 {
                // Probes come from the development split only.
                let prepared = prepare::prepare(&lm, false)?;
                items.extend(calibration::probes(&prepared.dataset, probes_per_type));
            }
            if let Some(path) = labels {
                items.extend(calibration::read_items(&std::fs::read_to_string(path)?)?);
            }
            anyhow::ensure!(!items.is_empty(), "no calibration items");
            let r = calibration::calibrate(
                judge.as_ref(),
                &items,
                lm.manifest.models.judge.max_output_tokens,
            )
            .await;
            if json {
                print_json(&r)?;
            } else {
                print!("{}", calibration::markdown(&r));
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Spend { action, spend } => {
            let ledger = spend.ledger()?;
            match action {
                SpendAction::Show { json } => {
                    let status = ledger.status()?;
                    if json {
                        print_json(&status)?;
                    } else {
                        print!("{}", keep_bench::spend::markdown(&status));
                    }
                }
                SpendAction::Settle { id, usd, note } => {
                    ledger.settle(&id, usd, note)?;
                    print!("{}", keep_bench::spend::markdown(&ledger.status()?));
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::AblationGrid => {
            #[derive(serde::Serialize)]
            struct Arms {
                arms: Vec<ablation::ArmSpec>,
            }
            print!(
                "{}",
                toml::to_string(&Arms {
                    arms: ablation::keep_ablation_grid()
                })?
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    anyhow::ensure!(
        !path.exists(),
        "{} already exists; frozen files are never overwritten",
        path.display()
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
