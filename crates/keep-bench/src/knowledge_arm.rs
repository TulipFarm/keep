//! The Knowledge arms (K0 and K1) of the Keep ablation.
//!
//! Each benchmark history becomes one Knowledge corpus: one source per
//! session and one episode per turn, ingested in date order with the clock
//! set to the session date, so "what was known when the question was asked"
//! is an ordinary as-of read. K0 answers from a hybrid search under the
//! retained-evidence allowance; K1 adds the level-0 table of contents in
//! the standing-context slot.
//!
//! The combined arms reuse both halves: Mk0 is the Memory context
//! ([`crate::memory_arm`]) followed by K0's evidence; Mk1 also hands the
//! level-0 table of contents to the Memory context's Knowledge slot through
//! its callback, so the whole standing context stays within
//! `standing_tokens` and evidence within `evidence_tokens`.
//!
//! The ranked arms use a second engine over the same storage, with
//! visibility-first BM25 fused with dense search ([`LexicalMode::Bm25`]) and
//! an optional cross-encoder reranker. K2 fills `evidence_tokens` from that
//! ranking; Mk2 ranks Memory entries (by BM25 over their lines and by dense
//! similarity of their source turns) and Knowledge results together and fills
//! `standing_tokens + evidence_tokens`, the same budget as the 6K retrieval
//! baselines. Both deduplicate by source turn and prefer whole source turns
//! over derived text, then present what they kept in conversation order.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use keep_knowledge::engine::{EpisodeInput, SourceInput};
use keep_knowledge::fusion::rrf;
use keep_knowledge::session_extract::SessionExtractor;
use keep_knowledge::session_summary::SessionSummarizer;
use keep_knowledge::subject::Grounding;
use keep_knowledge::time_ref::{TimeWindow, days_before, question_window};
use keep_knowledge::{
    Access, AsOf, EvidenceAllowance, Hit, IndexRequest, ItemRef, Knowledge, KnowledgeConfig,
    LexicalMode, MAX_EVIDENCE_TOKENS, ManualClock, SearchRequest, SourceKind, profile,
};
use keep_memory::{ContextBudget, KnowledgeToc};
use keep_providers::{Embedder, Reranker, Tokenizer, timing};
use keep_store::{Bm25Index, Bm25Params, LexicalIndex};

use crate::ablation::{FactOptions, KeepPath, Representation};
use crate::baselines::Built;
use crate::context::{
    BlockTokens, CONTEXT_FORMAT, ContextBundle, Slot, Truncation, render_full_history, render_turn,
};
use crate::dataset::{History, Query};
use crate::manifest::{RetrievalSpec, StorageProfile};
use crate::memory_arm::{MemoryArm, toc_block_tokens};

pub(crate) const PARTITION: &str = "default";
pub(crate) const COLLECTION: &str = "conversations";
/// The table of contents never takes more than this, even when the
/// standing allowance is larger.
const TOC_TOKENS: usize = 640;
const SEARCH_LIMIT: usize = 20;
/// The ranked arms ask for as many results as the 6K retrieval baselines
/// (`top_k = 200`), from pools four times that size, before filling.
const RANKED_LIMIT: usize = 200;
const RANKED_POOL: usize = 4 * RANKED_LIMIT;

struct Evidence {
    text: String,
    tokens: usize,
    ids: Vec<String>,
    turns: BTreeSet<String>,
    omitted: usize,
    method: String,
}

pub struct KnowledgeArms {
    k: Knowledge,
    clock: Arc<ManualClock>,
    tokenizer: Arc<dyn Tokenizer>,
    retrieval: RetrievalSpec,
    run_tag: String,
    /// History id to corpus, for histories already ingested.
    corpora: HashMap<String, String>,
    /// The ranked arms' engine: same storage, visibility-first hybrid search.
    ranked: Knowledge,
    reranker: Option<(Arc<dyn Reranker>, usize)>,
    /// The last ranked search, shared by K2 and Mk2 on the same question.
    last: Option<RankedSearch>,
    embedder: Arc<dyn Embedder>,
    /// Source-turn vectors for Mk2's dense Memory ranking, by text.
    turn_vectors: HashMap<String, Arc<[f32]>>,
    /// Extracted session facts for K3 and Mk3, when enabled.
    facts: Option<FactsState>,
}

struct FactsState {
    extractor: Arc<SessionExtractor>,
    facts_tokens: usize,
    profile_tokens: usize,
    /// History id to its extracted preferences, oldest session first.
    preferences: HashMap<String, Vec<crate::facts::Preference>>,
    /// History id to facts corpus, for histories already built.
    corpora: HashMap<String, String>,
    /// Facts corpus to its facts and subject names, loaded on first use.
    loaded: HashMap<String, Arc<LoadedFacts>>,
    /// Fact statement vectors, by text.
    vectors: HashMap<String, Arc<[f32]>>,
    /// Sessions with no cached extraction, over every facts corpus built.
    missing: usize,
    /// Session summaries for arms with `summaries`, when enabled.
    summarizer: Option<Arc<SessionSummarizer>>,
    summary_tokens: usize,
    /// History id to its sessions' cached summaries, oldest first.
    summaries: HashMap<String, Arc<Vec<crate::facts::SummaryLine>>>,
    source_times: HashMap<String, HashMap<String, crate::v5::SourceSnippet>>,
}

struct LoadedFacts {
    facts: Vec<keep_knowledge::model::Fact>,
    names: HashMap<String, String>,
}

struct RankedSearch {
    key: (String, String, Option<chrono::DateTime<chrono::Utc>>),
    hits: Vec<Hit>,
    method: String,
    /// Fact id to the turns its evidence cites, filled on first use.
    cites: HashMap<String, Vec<String>>,
}

/// One candidate for a ranked context.
struct Unit {
    /// Reported id: a Knowledge point id, or `turn:<id>` for a Memory entry
    /// upgraded to its source turn.
    id: String,
    /// Fusion key: whole source turns share `turn:<id>` whichever side
    /// found them, so agreement between Memory and Knowledge counts.
    key: String,
    text: String,
    /// Shorter text to use when `text` does not fit.
    fallback: Option<String>,
    kind: UnitKind,
}

enum UnitKind {
    /// The whole source turn.
    Turn(String),
    /// Part of a long turn, starting at this character.
    Part(String, i32),
    Fact,
}

/// Where Knowledge keeps its ledger and vectors for a run.
pub enum KnowledgeStorage {
    InMemory,
    PostgresVector {
        url: String,
    },
    PostgresQdrant {
        postgres_url: String,
        qdrant_url: String,
        api_key: Option<String>,
    },
}

impl KnowledgeStorage {
    pub fn profile(&self) -> StorageProfile {
        match self {
            KnowledgeStorage::InMemory => StorageProfile::InMemory,
            KnowledgeStorage::PostgresVector { .. } => StorageProfile::PostgresVector,
            KnowledgeStorage::PostgresQdrant { .. } => StorageProfile::PostgresQdrant,
        }
    }
}

impl KnowledgeArms {
    pub async fn new(
        storage: KnowledgeStorage,
        embedder: Arc<dyn Embedder>,
        tokenizer: Arc<dyn Tokenizer>,
        retrieval: RetrievalSpec,
        run_tag: &str,
        collection: String,
        reranker: Option<(Arc<dyn Reranker>, usize)>,
    ) -> anyhow::Result<Self> {
        let k = match &storage {
            KnowledgeStorage::InMemory => profile::in_memory(),
            KnowledgeStorage::PostgresVector { url } => profile::postgres_vector(url).await?,
            KnowledgeStorage::PostgresQdrant {
                postgres_url,
                qdrant_url,
                api_key,
            } => profile::postgres_qdrant(postgres_url, qdrant_url, api_key.clone()).await?,
        };
        let clock = Arc::new(ManualClock::new(chrono::DateTime::UNIX_EPOCH));
        let config = KnowledgeConfig {
            collection,
            ..KnowledgeConfig::default()
        };
        let k = k
            .with_clock(clock.clone())
            .with_embedder(embedder.clone())
            .with_tokenizer(tokenizer.clone())
            .with_config(config.clone());
        let mut ranked = k
            .sharing_storage(clock.clone())
            .with_config(KnowledgeConfig {
                candidate_pool: RANKED_POOL,
                lexical: LexicalMode::Bm25(Bm25Params {
                    k1: retrieval.bm25_k1,
                    b: retrieval.bm25_b,
                }),
                ..config
            });
        if let Some((r, top_n)) = &reranker {
            ranked = ranked.with_reranker(r.clone(), *top_n);
        }
        Ok(Self {
            k,
            clock,
            tokenizer,
            retrieval,
            run_tag: run_tag.into(),
            corpora: HashMap::new(),
            ranked,
            reranker,
            last: None,
            embedder,
            turn_vectors: HashMap::new(),
            facts: None,
        })
    }

    /// Let K3 and Mk3 run, reading session facts through `extractor`'s
    /// cache. Warm the cache ([`crate::facts::warm_cache`]) first; sessions
    /// it lacks contribute no facts and are counted.
    pub fn enable_facts(
        &mut self,
        extractor: Arc<SessionExtractor>,
        facts_tokens: usize,
        profile_tokens: usize,
    ) {
        self.facts = Some(FactsState {
            extractor,
            facts_tokens,
            profile_tokens,
            preferences: HashMap::new(),
            corpora: HashMap::new(),
            loaded: HashMap::new(),
            vectors: HashMap::new(),
            missing: 0,
            summarizer: None,
            summary_tokens: 0,
            summaries: HashMap::new(),
            source_times: HashMap::new(),
        });
    }

    /// Let fact arms with `summaries` read per-session summaries through
    /// `summarizer`'s cache, within `summary_tokens`. Call after
    /// [`Self::enable_facts`] and warm the cache first; sessions it lacks
    /// have no summary.
    pub fn enable_summaries(&mut self, summarizer: Arc<SessionSummarizer>, summary_tokens: usize) {
        if let Some(state) = self.facts.as_mut() {
            state.summarizer = Some(summarizer);
            state.summary_tokens = summary_tokens;
        }
    }

    /// Sessions that had no cached extraction in the facts corpora built.
    pub fn facts_missing(&self) -> usize {
        self.facts.as_ref().map_or(0, |f| f.missing)
    }

    fn facts_corpus_name(&self, history: &History) -> String {
        format!("bench-{}-{}-x3", self.run_tag, history.id)
    }

    /// Build the facts corpus of every history not built yet, up to
    /// `concurrency` at a time. Failures are left for first use to retry.
    pub async fn prepare_facts(
        &mut self,
        histories: &[&History],
        concurrency: usize,
    ) -> Vec<String> {
        let Some(state) = self.facts.as_ref() else {
            return Vec::new();
        };
        let gate = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        let mut queued = BTreeSet::new();
        for h in histories {
            if state.corpora.contains_key(&h.id) || !queued.insert(h.id.clone()) {
                continue;
            }
            let clock = Arc::new(ManualClock::new(chrono::DateTime::UNIX_EPOCH));
            let k = Knowledge::new(self.k.ledger().clone(), None).with_clock(clock.clone());
            let corpus = self.facts_corpus_name(h);
            let (history, gate) = ((*h).clone(), gate.clone());
            let extractor = state.extractor.clone();
            tasks.spawn(async move {
                let _permit = gate.acquire_owned().await.expect("semaphore open");
                let r = crate::facts::ingest_facts(&k, &clock, &corpus, &history, &extractor).await;
                (history.id, corpus, r)
            });
        }
        let mut errors = Vec::new();
        let state = self.facts.as_mut().expect("checked above");
        while let Some(done) = tasks.join_next().await {
            match done {
                Ok((id, corpus, Ok(x))) => {
                    state.missing += x.missing;
                    state.preferences.insert(id.clone(), x.preferences);
                    state.source_times.insert(id.clone(), x.source_times);
                    state.corpora.insert(id, corpus);
                }
                Ok((id, _, Err(e))) => errors.push(format!("facts ingest of {id}: {e}")),
                Err(e) => errors.push(format!("facts ingest task: {e}")),
            }
        }
        errors
    }

    async fn facts_corpus(&mut self, history: &History) -> anyhow::Result<String> {
        let name = self.facts_corpus_name(history);
        let state = self
            .facts
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("extracted facts are not enabled for this run"))?;
        if let Some(c) = state.corpora.get(&history.id) {
            return Ok(c.clone());
        }
        let clock = Arc::new(ManualClock::new(chrono::DateTime::UNIX_EPOCH));
        let k = Knowledge::new(self.k.ledger().clone(), None).with_clock(clock.clone());
        let x = crate::facts::ingest_facts(&k, &clock, &name, history, &state.extractor).await?;
        state.missing += x.missing;
        state.preferences.insert(history.id.clone(), x.preferences);
        state
            .source_times
            .insert(history.id.clone(), x.source_times);
        state.corpora.insert(history.id.clone(), name.clone());
        Ok(name)
    }

    async fn load_facts(&mut self, corpus: &str) -> anyhow::Result<Arc<LoadedFacts>> {
        let state = self.facts.as_mut().expect("facts enabled");
        if let Some(l) = state.loaded.get(corpus) {
            return Ok(l.clone());
        }
        let ledger = self.k.ledger();
        let mut ids: Vec<String> = ledger
            .point_ids(corpus)
            .await?
            .into_iter()
            .filter(|id| id.starts_with("fact_"))
            .collect();
        ids.sort();
        let mut facts = ledger.facts(corpus, &ids).await?;
        facts.sort_by(|a, b| a.id.cmp(&b.id));
        let mut names = HashMap::new();
        for f in &facts {
            if !names.contains_key(&f.subject) {
                let name = ledger
                    .entity(corpus, &f.subject)
                    .await?
                    .map_or_else(|| f.subject.clone(), |e| e.name);
                names.insert(f.subject.clone(), name);
            }
        }
        let loaded = Arc::new(LoadedFacts { facts, names });
        state.loaded.insert(corpus.to_owned(), loaded.clone());
        Ok(loaded)
    }

    /// The dated facts block for `query`: facts known when it was asked,
    /// ranked by BM25 and dense similarity fused with RRF, grouped by
    /// subject and attribute, whole groups taken in rank order within
    /// `budget`.
    ///
    /// With `opts.temporal` and a time reference in the question, facts
    /// that started inside its window are a third RRF list, every line
    /// shows its age at the question, and the block opens with the window.
    /// With `opts.preferences`, facts the preference profile shows are
    /// left out.
    async fn facts_block(
        &mut self,
        history: &History,
        query: &Query,
        budget: usize,
        opts: FactOptions,
        window: Option<&TimeWindow>,
        prune: Option<&FactPrune>,
    ) -> anyhow::Result<(crate::facts::FactsBlock, String)> {
        use crate::facts::{FactLine, assemble, group_lines, rank_groups};
        let corpus = self.facts_corpus(history).await?;
        let loaded = self.load_facts(&corpus).await?;
        let as_of = query.asked_at.map_or_else(AsOf::now, AsOf::known_at);
        let asked = query.asked_at.map(|t| t.date_naive());
        let shown_in_profile: BTreeSet<(String, String)> = if opts.preferences && !opts.provenance {
            let state = self.facts.as_ref().expect("facts enabled");
            state
                .preferences
                .get(&history.id)
                .into_iter()
                .flatten()
                .map(crate::facts::Preference::fact_key)
                .collect()
        } else {
            BTreeSet::new()
        };
        let mut lines: Vec<FactLine> = loaded
            .facts
            .iter()
            .filter(|f| !shown_in_profile.contains(&(f.predicate.clone(), f.object.clone())))
            .filter_map(|f| {
                let v = keep_knowledge::temporal::fact_as_of(f, as_of)?;
                Some(FactLine {
                    id: f.id.clone(),
                    subject: loaded
                        .names
                        .get(&f.subject)
                        .cloned()
                        .unwrap_or_else(|| f.subject.clone()),
                    predicate: f.predicate.clone(),
                    value: f.object.clone(),
                    text: f.text.clone(),
                    valid_from: v.version.valid_from.map(|t| t.date_naive()),
                    valid_to: v.version.valid_to.map(|t| t.date_naive()),
                    basis: v.version.valid_from_basis,
                    status: v.version.status,
                    turns: v
                        .evidence
                        .iter()
                        .map(|e| e.episode_id.clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    age_days: match (opts.temporal, v.version.valid_from, asked) {
                        (true, Some(from), Some(asked)) => {
                            Some(days_before(from.date_naive(), asked))
                        }
                        _ => None,
                    },
                })
            })
            .collect();
        if let Some(p) = prune {
            // Attributes whose value changed keep their whole history.
            let changed: BTreeSet<(String, String)> = lines
                .iter()
                .filter(|l| l.status != keep_knowledge::model::FactStatus::Active)
                .map(|l| (l.subject.clone(), l.predicate.clone()))
                .collect();
            lines.retain(|l| {
                p.keeps(
                    l,
                    changed.contains(&(l.subject.clone(), l.predicate.clone())),
                )
            });
        }
        let lexical: Vec<String> = Bm25Index::build(
            Bm25Params {
                k1: self.retrieval.bm25_k1,
                b: self.retrieval.bm25_b,
            },
            lines.iter().map(|l| (l.id.clone(), l.search_text())),
        )
        .search(&query.text, lines.len().max(1))
        .into_iter()
        .map(|h| h.id)
        .collect();
        let dense = self.dense_facts(&lines, &query.text).await?;
        let rrf_k = self.ranked.config().rrf_k;
        let mut lists = vec![lexical, dense];
        if opts.decompose {
            let index = Bm25Index::build(
                Bm25Params {
                    k1: self.retrieval.bm25_k1,
                    b: self.retrieval.bm25_b,
                },
                lines.iter().map(|l| (l.id.clone(), l.search_text())),
            );
            for facet in crate::v5::facets(&query.text, &participants(history))
                .iter()
                .skip(1)
            {
                lists.push(
                    index
                        .search(facet, lines.len().max(1))
                        .into_iter()
                        .map(|h| h.id)
                        .collect(),
                );
                lists.push(self.dense_facts(&lines, facet).await?);
            }
        }
        let mut opening = String::new();
        let people = participants(history);
        let grounding = if opts.subjects {
            Grounding::of(&query.text, &people)
        } else {
            None
        };
        if let Some(g) = &grounding {
            opening = format!("{}\n", crate::facts::grounding_line(g));
        }
        if let Some(w) = window {
            // In BM25 order, so the window list still prefers relevant facts.
            let inside: Vec<String> = lists[0]
                .iter()
                .filter(|id| {
                    lines
                        .iter()
                        .any(|l| &l.id == *id && l.valid_from.is_some_and(|d| w.contains(d)))
                })
                .cloned()
                .collect();
            lists.push(inside);
            opening.push_str(&format!("{}\n", crate::facts::window_line(w)));
        }
        let mut fused: Vec<String> = rrf(&lists, rrf_k).into_iter().map(|(id, _)| id).collect();
        let mut expanded = 0;
        if opts.expand {
            let extra = crate::facts::expansion(&lines, &fused, &lists[0], &people);
            expanded = extra.len();
            if !extra.is_empty() {
                lists.push(extra);
                fused = rrf(&lists, rrf_k).into_iter().map(|(id, _)| id).collect();
            }
        }
        if opts.fact_selection
            && let Some((reranker, top_n)) = &self.reranker
        {
            let head = fused.len().min(*top_n);
            let by_id: HashMap<_, _> = lines.iter().map(|l| (&l.id, l)).collect();
            let texts: Vec<_> = fused[..head]
                .iter()
                .map(|id| by_id[id].search_text())
                .collect();
            if head > 1 {
                let scores = reranker.score(&query.text, &texts).await?;
                let mut indices: Vec<_> = (0..head).collect();
                indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                let reordered: Vec<_> = indices.iter().map(|&i| fused[i].clone()).collect();
                fused.splice(..head, reordered);
            }
        }
        let other_facts = grounding
            .as_ref()
            .map(|g| crate::v5::other_fact_ids(&lines, g))
            .unwrap_or_default();
        let other = |group: &crate::facts::FactGroup| {
            group
                .lines
                .iter()
                .all(|line| other_facts.contains(&line.id))
        };
        let state = self.facts.as_mut().expect("facts enabled");
        let annotations = &state.source_times[&history.id];
        if opts.provenance || opts.relative_dates {
            for line in &mut lines {
                crate::v5::attach_sources(line, annotations, opts.provenance, opts.relative_dates);
            }
        }
        let mut ranked = if opts.fact_selection {
            crate::v5::individual_groups(lines, &fused)
        } else {
            rank_groups(group_lines(lines), &fused)
        };
        if opts.fact_selection && grounding.is_some() {
            ranked.sort_by_key(other);
        }
        let opening_tokens = if opening.is_empty() {
            0
        } else {
            self.tokenizer.count(&opening) + 1
        };
        let mut block = match &grounding {
            Some(g) => crate::facts::assemble_grounded(
                &ranked,
                other,
                &crate::facts::other_heading(g),
                budget.saturating_sub(opening_tokens),
                self.tokenizer.as_ref(),
            ),
            None => assemble(
                &ranked,
                budget.saturating_sub(opening_tokens),
                self.tokenizer.as_ref(),
            ),
        };
        if !opening.is_empty() && !block.text.is_empty() {
            block.text = format!("{opening}{}", block.text);
            block.tokens = self.tokenizer.count(&block.text);
        }
        let method = format!(
            "facts=extract({}) as_of=asked_at rank=rrf(k={rrf_k};bm25(k1={},b={}),dense({}:{}){}) group=subject+attribute{} budget={budget}",
            state.extractor.evidence_label(),
            self.retrieval.bm25_k1,
            self.retrieval.bm25_b,
            self.embedder.descriptor().provider,
            self.embedder.descriptor().model,
            if window.is_some() { ",window" } else { "" },
            match (opts.temporal, opts.preferences) {
                (true, true) => " temporal profile",
                (true, false) => " temporal",
                (false, true) => " profile",
                (false, false) => "",
            },
        );
        let mut method = method;
        if opts.provenance || opts.relative_dates || opts.fact_selection || opts.decompose {
            method.push_str(&format!(
                " v5(provenance={},relative_dates={},fact_selection={},decompose={})",
                opts.provenance, opts.relative_dates, opts.fact_selection, opts.decompose
            ));
        }
        if opts.subjects {
            method.push_str(&match &grounding {
                Some(g) => format!(
                    " subjects(asked={};others={})",
                    g.asked_label(),
                    g.others_label()
                ),
                None => " subjects(none)".into(),
            });
        }
        if opts.expand {
            method.push_str(&format!(
                " expand(seeds={},added={expanded})",
                crate::facts::EXPAND_SEEDS
            ));
        }
        Ok((block, method))
    }

    /// The cached summaries of `history`'s sessions, oldest first; sessions
    /// with no summary, or one with nothing about the people, are left out.
    fn session_summaries(&mut self, history: &History) -> Arc<Vec<crate::facts::SummaryLine>> {
        let state = self.facts.as_mut().expect("facts enabled");
        if let Some(s) = state.summaries.get(&history.id) {
            return s.clone();
        }
        let mut lines: Vec<crate::facts::SummaryLine> = match &state.summarizer {
            Some(summarizer) => history
                .sessions
                .iter()
                .filter_map(|s| {
                    let text = summarizer.cached(&crate::facts::session_input(s))?;
                    (!text.is_empty()).then(|| crate::facts::SummaryLine {
                        session: s.id.clone(),
                        date: s.date.map(|d| d.date_naive()),
                        text,
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        lines.sort_by_key(|l| l.date);
        let lines = Arc::new(lines);
        state.summaries.insert(history.id.clone(), lines.clone());
        lines
    }

    /// The most relevant session summaries for `query` within `budget`:
    /// sessions on or before the question date, ranked by BM25 and dense
    /// similarity (and the question's time window, when there is one)
    /// fused with RRF.
    async fn summaries_block(
        &mut self,
        history: &History,
        query: &Query,
        budget: usize,
        window: Option<&TimeWindow>,
    ) -> anyhow::Result<(crate::facts::SummariesBlock, String)> {
        let all = self.session_summaries(history);
        let asked = query.asked_at.map(|t| t.date_naive());
        let known: Vec<usize> = (0..all.len())
            .filter(|&i| match (all[i].date, asked) {
                (Some(d), Some(a)) => d <= a,
                _ => true,
            })
            .collect();
        let lexical: Vec<String> = Bm25Index::build(
            Bm25Params {
                k1: self.retrieval.bm25_k1,
                b: self.retrieval.bm25_b,
            },
            known.iter().map(|&i| (i.to_string(), all[i].text.clone())),
        )
        .search(&query.text, known.len().max(1))
        .into_iter()
        .map(|h| h.id)
        .collect();
        let as_lines: Vec<crate::facts::FactLine> = known
            .iter()
            .map(|&i| crate::facts::FactLine {
                id: i.to_string(),
                subject: String::new(),
                predicate: String::new(),
                value: String::new(),
                text: all[i].text.clone(),
                valid_from: all[i].date,
                valid_to: None,
                basis: keep_knowledge::model::TimeBasis::Explicit,
                status: keep_knowledge::model::FactStatus::Active,
                turns: Vec::new(),
                age_days: None,
            })
            .collect();
        let dense = self.dense_facts(&as_lines, &query.text).await?;
        let mut lists = vec![lexical, dense];
        if let Some(w) = window {
            let inside: Vec<String> = lists[0]
                .iter()
                .filter(|id| {
                    id.parse::<usize>()
                        .ok()
                        .and_then(|i| all[i].date)
                        .is_some_and(|d| w.contains(d))
                })
                .cloned()
                .collect();
            lists.push(inside);
        }
        let rrf_k = self.ranked.config().rrf_k;
        let ranked: Vec<usize> = rrf(&lists, rrf_k)
            .into_iter()
            .filter_map(|(id, _)| id.parse().ok())
            .collect();
        let block =
            crate::facts::assemble_summaries(&all, &ranked, budget, self.tokenizer.as_ref());
        let method = format!(
            "summaries=summarize({}) rank=rrf(k={rrf_k};bm25,dense{}) {} of {} sessions budget={budget}",
            self.facts
                .as_ref()
                .and_then(|s| s.summarizer.as_ref())
                .map_or_else(|| "none".into(), |s| s.label()),
            if window.is_some() { ",window" } else { "" },
            block.sessions.len(),
            known.len(),
        );
        Ok((block, method))
    }

    /// The Memory preference profile for `query`: the history's extracted
    /// preferences from sessions on or before the question date.
    async fn preference_block(
        &self,
        history: &History,
        query: &Query,
        budget: usize,
    ) -> anyhow::Result<crate::memory_arm::ProfileBlock> {
        let state = self.facts.as_ref().expect("facts enabled");
        let lines: Vec<crate::memory_arm::ProfileLine> = state
            .preferences
            .get(&history.id)
            .into_iter()
            .flatten()
            .filter(|p| match (p.said, query.asked_at) {
                (Some(said), Some(asked)) => said <= asked,
                _ => true,
            })
            .map(|p| crate::memory_arm::ProfileLine {
                date: p.said,
                text: p.statement.clone(),
                turns: p.turns.clone(),
            })
            .collect();
        crate::memory_arm::preference_profile(&lines, budget, query, self.tokenizer.as_ref())
            .await
            .map_err(anyhow::Error::msg)
    }

    /// Fact ids by the cosine similarity of their statement to the
    /// question, best first, ties by id. Each statement is embedded once.
    async fn dense_facts(
        &mut self,
        lines: &[crate::facts::FactLine],
        question: &str,
    ) -> anyhow::Result<Vec<String>> {
        let state = self.facts.as_mut().expect("facts enabled");
        let missing: Vec<String> = lines
            .iter()
            .filter(|l| !state.vectors.contains_key(&l.text))
            .map(|l| l.text.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for batch in missing.chunks(64) {
            let _t = timing::span("k3.fact_embed").items(batch.len());
            let vectors = self.embedder.embed(batch).await?.vectors;
            for (text, v) in batch.iter().zip(vectors) {
                state.vectors.insert(text.clone(), v.into());
            }
        }
        let q = self
            .embedder
            .embed(&[question.to_owned()])
            .await?
            .vectors
            .into_iter()
            .next()
            .unwrap_or_default();
        let mut scored: Vec<(f32, &str)> = lines
            .iter()
            .map(|l| (cosine(&q, &state.vectors[&l.text]), l.id.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        Ok(scored.into_iter().map(|(_, id)| id.to_owned()).collect())
    }

    pub fn vectors_capabilities(&self) -> Option<keep_store::VectorCapabilities> {
        self.k.vectors().map(|v| v.capabilities())
    }

    fn corpus_name(&self, history: &History) -> String {
        format!("bench-{}-{}", self.run_tag, history.id)
    }

    async fn corpus(&mut self, history: &History) -> anyhow::Result<String> {
        if let Some(c) = self.corpora.get(&history.id) {
            return Ok(c.clone());
        }
        let corpus = self.corpus_name(history);
        ingest_history(&self.k, &self.clock, &corpus, history).await?;
        self.corpora.insert(history.id.clone(), corpus.clone());
        Ok(corpus)
    }

    /// Ingest every history not ingested yet, up to `concurrency` at a
    /// time. Corpora are independent and each worker has its own clock, so
    /// the result is what one-at-a-time ingestion leaves. A history that
    /// fails here is left for [`Self::corpus`] to retry and report.
    pub async fn prepare(&mut self, histories: &[&History], concurrency: usize) -> Vec<String> {
        let gate = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        let mut queued = BTreeSet::new();
        for h in histories {
            if self.corpora.contains_key(&h.id) || !queued.insert(h.id.clone()) {
                continue;
            }
            let clock = Arc::new(ManualClock::new(chrono::DateTime::UNIX_EPOCH));
            let k = self.k.sharing_storage(clock.clone());
            let corpus = self.corpus_name(h);
            let (history, gate) = ((*h).clone(), gate.clone());
            tasks.spawn(async move {
                let _permit = gate.acquire_owned().await.expect("semaphore open");
                let r = ingest_history(&k, &clock, &corpus, &history).await;
                (history.id, corpus, r)
            });
        }
        let mut errors = Vec::new();
        while let Some(done) = tasks.join_next().await {
            match done {
                Ok((id, corpus, Ok(()))) => {
                    self.corpora.insert(id, corpus);
                }
                Ok((id, _, Err(e))) => errors.push(format!("knowledge ingest of {id}: {e}")),
                Err(e) => errors.push(format!("knowledge ingest task: {e}")),
            }
        }
        errors
    }

    pub async fn build(&mut self, path: KeepPath, history: &History, query: &Query) -> Built {
        match self.try_build(path, history, query).await {
            Ok(b) => Built::Ready(b),
            Err(e) => Built::Failed(format!("knowledge arm: {e}")),
        }
    }

    /// The level-0 table of contents cut to `budget` tokens.
    async fn toc(&self, corpus: &str, budget: usize) -> anyhow::Result<String> {
        let _t = timing::span("knowledge.toc").items(1);
        let view = self
            .k
            .index(
                corpus,
                &Access::to([PARTITION]),
                &IndexRequest {
                    level: 0,
                    budget_tokens: Some(budget),
                    ..Default::default()
                },
            )
            .await?;
        Ok(view.text)
    }

    /// Hybrid search under the retained-evidence allowance.
    async fn evidence(&self, corpus: &str, query: &Query) -> anyhow::Result<Evidence> {
        let access = Access::to([PARTITION]);
        let as_of = query.asked_at.map_or_else(AsOf::now, AsOf::known_at);
        let mut allowance = EvidenceAllowance::new(self.retrieval.evidence_tokens);
        let res = self
            .k
            .search(
                &SearchRequest {
                    corpus: corpus.into(),
                    access: access.clone(),
                    query: query.text.clone(),
                    as_of,
                    limit: SEARCH_LIMIT,
                },
                &mut allowance,
            )
            .await?;
        let _t = timing::span("knowledge.cite").items(res.hits.len());
        let mut turns = BTreeSet::new();
        let mut text = String::new();
        for h in &res.hits {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&h.text);
            turns.insert(h.citation.episode_id.clone());
            if let ItemRef::Fact(_) = h.item {
                // A fact carries all of its supporting turns.
                for c in self.k.cite(corpus, &access, &h.item.point_id()).await? {
                    turns.insert(c.episode_id);
                }
            }
        }
        Ok(Evidence {
            tokens: self.tokenizer.count(&text),
            text,
            ids: res.hits.iter().map(|h| h.item.point_id()).collect(),
            turns,
            omitted: res.omitted,
            method: res.method,
        })
    }

    fn evidence_block(&self, e: &Evidence) -> BlockTokens {
        BlockTokens::new(
            "knowledge-evidence",
            Slot::Evidence,
            e.tokens,
            self.retrieval.evidence_tokens + e.ids.len(),
        )
    }

    async fn try_build(
        &mut self,
        path: KeepPath,
        history: &History,
        query: &Query,
    ) -> anyhow::Result<ContextBundle> {
        let corpus = self.corpus(history).await?;
        let (_, visibility) = render_full_history(history, query.asked_at);

        let mut text = String::new();
        let mut standing = 0;
        let mut blocks = Vec::new();
        if path.uses_toc() {
            let budget = TOC_TOKENS.min(self.retrieval.standing_tokens);
            text = self.toc(&corpus, budget).await?;
            standing = budget;
            blocks.push(BlockTokens::new(
                "knowledge-toc",
                Slot::Standing,
                self.tokenizer.count(&text),
                budget,
            ));
        }

        let ev = self.evidence(&corpus, query).await?;
        if !text.is_empty() && !ev.text.is_empty() {
            text.push('\n');
        }
        text.push_str(&ev.text);
        blocks.push(self.evidence_block(&ev));
        let tokens = self.tokenizer.count(&text);
        Ok(ContextBundle {
            format: CONTEXT_FORMAT.into(),
            text,
            tokens,
            budget_tokens: Some(standing + self.retrieval.evidence_tokens + ev.ids.len()),
            method: Some(format!(
                "keep-knowledge;{};toc={}",
                ev.method,
                path.uses_toc()
            )),
            retrieved_ids: ev.ids.clone(),
            included_ids: ev.ids,
            included_turn_ids: ev.turns.into_iter().collect(),
            visibility,
            truncation: Some(Truncation {
                reason: "retained evidence allowance".into(),
                dropped_chunks: ev.omitted,
                dropped_turns: 0,
                dropped_tokens: 0,
            }),
            blocks,
        })
    }

    /// The combined arms. Mk0 is the Memory context plus Knowledge evidence;
    /// Mk1 also offers the level-0 table of contents to the Memory context's
    /// Knowledge slot, so the TOC is charged to the standing budget and the
    /// Memory files get whatever it leaves unused.
    pub async fn build_with_memory(
        &mut self,
        path: KeepPath,
        representation: Representation,
        memory: &mut MemoryArm,
        history: &History,
        query: &Query,
    ) -> Built {
        match self
            .try_build_with_memory(path, representation, memory, history, query)
            .await
        {
            Ok(b) => b,
            Err(e) => Built::Failed(format!("memory+knowledge arm: {e}")),
        }
    }

    async fn try_build_with_memory(
        &mut self,
        path: KeepPath,
        representation: Representation,
        memory: &mut MemoryArm,
        history: &History,
        query: &Query,
    ) -> anyhow::Result<Built> {
        let standing_tokens = self.retrieval.standing_tokens;
        let slot = match ContextBudget::for_total(standing_tokens) {
            Ok(b) => b.knowledge,
            Err(e) => return Ok(Built::Unavailable(e.to_string())),
        };
        let corpus = self.corpus(history).await?;
        let toc = if path.uses_toc() {
            self.toc_for_slot(&corpus, slot).await?
        } else {
            None
        };
        let offer = |_: usize| toc.clone();
        let toc_source: Option<&dyn KnowledgeToc> = toc.as_ref().map(|_| &offer as _);
        let mut bundle = match memory
            .build_with_toc(
                representation,
                standing_tokens,
                history,
                query,
                self.tokenizer.as_ref(),
                toc_source,
            )
            .await
        {
            Built::Ready(b) => b,
            other => return Ok(other),
        };
        if path.uses_toc() && bundle.blocks.iter().all(|b| b.name != "knowledge-toc") {
            bundle
                .blocks
                .push(BlockTokens::new("knowledge-toc", Slot::Standing, 0, slot));
        }

        let ev = self.evidence(&corpus, query).await?;
        let (_, visibility) = render_full_history(history, query.asked_at);
        if !ev.text.is_empty() {
            bundle.text.push('\n');
            bundle.text.push_str(&ev.text);
        }
        bundle.blocks.push(self.evidence_block(&ev));
        bundle.tokens = self.tokenizer.count(&bundle.text);
        bundle.budget_tokens =
            Some(standing_tokens + self.retrieval.evidence_tokens + ev.ids.len());
        bundle.method = Some(format!(
            "keep-mk;{};{};toc={}",
            bundle.method.as_deref().unwrap_or_default(),
            ev.method,
            path.uses_toc()
        ));
        bundle.retrieved_ids.extend(ev.ids.iter().cloned());
        bundle.included_ids.extend(ev.ids);
        let mut turns: BTreeSet<String> = bundle.included_turn_ids.drain(..).collect();
        turns.extend(ev.turns);
        bundle.included_turn_ids = turns.into_iter().collect();
        bundle.visibility = visibility;
        let dropped_turns = visibility
            .turns
            .saturating_sub(bundle.included_turn_ids.len());
        bundle.truncation = Some(match bundle.truncation.take() {
            Some(mut t) => {
                t.reason = format!("{}; retained evidence allowance", t.reason);
                t.dropped_chunks += ev.omitted;
                t.dropped_turns = dropped_turns;
                t
            }
            None => Truncation {
                reason: "retained evidence allowance".into(),
                dropped_chunks: ev.omitted,
                dropped_turns,
                dropped_tokens: 0,
            },
        });
        Ok(Built::Ready(bundle))
    }

    /// The largest level-0 table of contents whose block, framing included,
    /// fits the Memory context's Knowledge slot of `slot` tokens.
    async fn toc_for_slot(&self, corpus: &str, slot: usize) -> anyhow::Result<Option<String>> {
        let overhead = toc_block_tokens("", self.tokenizer.as_ref());
        let mut budget = slot.min(TOC_TOKENS).saturating_sub(overhead);
        // JSON escaping can push the block over; shrink by the overrun.
        for _ in 0..4 {
            if budget == 0 {
                break;
            }
            let text = self.toc(corpus, budget).await?;
            if text.trim().is_empty() {
                return Ok(None);
            }
            let cost = toc_block_tokens(&text, self.tokenizer.as_ref());
            if cost <= slot {
                return Ok(Some(text));
            }
            budget = budget.saturating_sub(cost - slot);
        }
        Ok(None)
    }

    /// K2 (no `memory`) or Mk2 (with it): one ranking over every candidate,
    /// filled to the arm's whole budget.
    pub async fn build_ranked(
        &mut self,
        path: KeepPath,
        representation: Representation,
        memory: Option<&mut MemoryArm>,
        history: &History,
        query: &Query,
        opts: FactOptions,
    ) -> Built {
        let built = if path.uses_extracted_facts() {
            self.try_build_with_facts(path, representation, memory, history, query, opts)
                .await
        } else {
            self.try_build_ranked(
                path,
                representation,
                memory,
                history,
                query,
                None,
                None,
                false,
                false,
            )
            .await
        };
        match built {
            Ok(b) => b,
            Err(e) => Built::Failed(format!("ranked keep arm: {e}")),
        }
    }

    /// The ranked engine's results for `query`, reusing the last search.
    async fn ranked_search(
        &mut self,
        corpus: &str,
        query: &Query,
    ) -> anyhow::Result<&mut RankedSearch> {
        let key = (corpus.to_owned(), query.text.clone(), query.asked_at);
        if self.last.as_ref().is_none_or(|l| l.key != key) {
            // The allowance only bounds the candidate list; the arm fills
            // its own budget after deduplication.
            let mut allowance = EvidenceAllowance::new(MAX_EVIDENCE_TOKENS);
            let res = self
                .ranked
                .search(
                    &SearchRequest {
                        corpus: corpus.into(),
                        access: Access::to([PARTITION]),
                        query: query.text.clone(),
                        as_of: query.asked_at.map_or_else(AsOf::now, AsOf::known_at),
                        limit: RANKED_LIMIT,
                    },
                    &mut allowance,
                )
                .await?;
            self.last = Some(RankedSearch {
                key,
                hits: res.hits,
                method: res.method,
                cites: HashMap::new(),
            });
        }
        Ok(self.last.as_mut().expect("just set"))
    }

    /// Memory entries ranked by the cosine similarity of their source turn
    /// to the question, best first, ties by key. Each source turn is
    /// embedded once per run.
    async fn dense_memory(
        &mut self,
        entries: &[crate::memory_arm::MemoryEntry],
        turns: &TurnIndex<'_>,
        question: &str,
    ) -> anyhow::Result<Vec<String>> {
        let mut texts: Vec<(String, &str)> = Vec::new();
        let mut seen = BTreeSet::new();
        for e in entries {
            if let Some(text) = turns.source(&e.turn_id)
                && seen.insert(e.turn_id.as_str())
            {
                texts.push((format!("turn:{}", e.turn_id), text));
            }
        }
        let missing: Vec<String> = texts
            .iter()
            .filter(|(_, t)| !self.turn_vectors.contains_key(*t))
            .map(|(_, t)| (*t).to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for batch in missing.chunks(64) {
            let _t = timing::span("mk2.memory_embed").items(batch.len());
            let vectors = self.embedder.embed(batch).await?.vectors;
            for (text, v) in batch.iter().zip(vectors) {
                self.turn_vectors.insert(text.clone(), v.into());
            }
        }
        let q = self
            .embedder
            .embed(&[question.to_owned()])
            .await?
            .vectors
            .into_iter()
            .next()
            .unwrap_or_default();
        let mut scored: Vec<(f32, String)> = texts
            .into_iter()
            .map(|(key, text)| (cosine(&q, &self.turn_vectors[text]), key))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(RANKED_LIMIT);
        Ok(scored.into_iter().map(|(_, k)| k).collect())
    }

    /// K2's or Mk2's ranked fill within `budget` (default: the arm's own),
    /// with units from inside `window` fused in as an extra list.
    #[allow(clippy::too_many_arguments)]
    async fn try_build_ranked(
        &mut self,
        path: KeepPath,
        representation: Representation,
        memory: Option<&mut MemoryArm>,
        history: &History,
        query: &Query,
        budget: Option<usize>,
        window: Option<&TimeWindow>,
        decompose: bool,
        focus: bool,
    ) -> anyhow::Result<Built> {
        let budget = budget.unwrap_or_else(|| self.ranked_budget(path));
        let corpus = self.corpus(history).await?;
        let (_, visibility) = render_full_history(history, query.asked_at);
        let turns = TurnIndex::new(history);
        let people = participants(history);
        // `focus`: search the excerpts for what is asked, not for the name.
        let focused = focus
            .then(|| crate::v8::focus_query(&query.text, &people))
            .flatten();
        let ranked_query = match &focused {
            Some(text) => Query {
                text: text.clone(),
                ..query.clone()
            },
            None => query.clone(),
        };
        let query_text = ranked_query.text.as_str();

        let search = self.ranked_search(&corpus, &ranked_query).await?;
        let mut units: Vec<Unit> = search
            .hits
            .iter()
            .map(|h| knowledge_unit(h, &turns))
            .collect();
        let search_method = search.method.clone();
        if decompose {
            let mut lists = vec![units.iter().map(|u| u.key.clone()).collect::<Vec<_>>()];
            let mut seen: BTreeSet<_> = units.iter().map(|u| u.key.clone()).collect();
            for facet in crate::v5::facets(&query.text, &participants(history))
                .iter()
                .skip(1)
            {
                let mut subquery = query.clone();
                subquery.text = facet.clone();
                let search = self.ranked_search(&corpus, &subquery).await?;
                let extra: Vec<_> = search
                    .hits
                    .iter()
                    .map(|h| knowledge_unit(h, &turns))
                    .collect();
                lists.push(extra.iter().map(|u| u.key.clone()).collect());
                units.extend(extra.into_iter().filter(|u| seen.insert(u.key.clone())));
            }
            let mut by_key: HashMap<_, _> = units.into_iter().map(|u| (u.key.clone(), u)).collect();
            units = rrf(&lists, self.ranked.config().rrf_k)
                .into_iter()
                .filter_map(|(id, _)| by_key.remove(&id))
                .collect();
        }
        let mut order: Vec<usize> = (0..units.len()).collect();

        let memory_method = match memory {
            Some(memory) => {
                let entries = match memory.entries(representation, history, query).await {
                    Ok(e) => e,
                    Err(e) => return Ok(Built::Failed(e)),
                };
                let index = Bm25Index::build(
                    Bm25Params {
                        k1: self.retrieval.bm25_k1,
                        b: self.retrieval.bm25_b,
                    },
                    entries
                        .iter()
                        .map(|e| (format!("turn:{}", e.turn_id), e.line.clone())),
                );
                let memory_list: Vec<String> = index
                    .search(query_text, RANKED_LIMIT)
                    .into_iter()
                    .map(|h| h.id)
                    .collect();
                let dense_list = self.dense_memory(&entries, &turns, query_text).await?;
                let by_key: HashMap<String, usize> = units
                    .iter()
                    .enumerate()
                    .map(|(i, u)| (u.key.clone(), i))
                    .collect();
                let knowledge_list: Vec<String> = units.iter().map(|u| u.key.clone()).collect();
                let lines: HashMap<&str, &str> = entries
                    .iter()
                    .map(|e| (e.turn_id.as_str(), e.line.as_str()))
                    .collect();
                let mut by_key = by_key;
                let mut seen = BTreeSet::new();
                for key in memory_list.iter().chain(&dense_list) {
                    if !seen.insert(key) {
                        continue;
                    }
                    let turn_id = &key["turn:".len()..];
                    let line = lines[turn_id];
                    match by_key.get(key) {
                        Some(&i) => units[i].fallback = Some(turns.excerpt(turn_id, line)),
                        None => {
                            by_key.insert(key.clone(), units.len());
                            units.push(memory_unit(turn_id, line, &turns));
                        }
                    }
                }
                let fused = rrf(
                    &[knowledge_list, memory_list, dense_list],
                    self.ranked.config().rrf_k,
                );
                order = fused.iter().map(|(key, _)| by_key[key]).collect();
                if let Some((reranker, top_n)) = &self.reranker {
                    let head = order.len().min(*top_n);
                    if head > 1 {
                        let texts: Vec<String> = order[..head]
                            .iter()
                            .map(|&i| units[i].text.clone())
                            .collect();
                        let scores = reranker.score(query_text, &texts).await?;
                        let mut idx: Vec<usize> = (0..head).collect();
                        // Stable: equal scores keep their fused order.
                        idx.sort_by(|&a, &b| {
                            scores[b]
                                .partial_cmp(&scores[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let reordered: Vec<usize> = idx.iter().map(|&j| order[j]).collect();
                        order.splice(..head, reordered);
                    }
                }
                format!(
                    "memory=bm25(k1={},b={})+dense({}:{}, source turns) over {} entries;joint=rrf(k={}){}",
                    self.retrieval.bm25_k1,
                    self.retrieval.bm25_b,
                    self.embedder.descriptor().provider,
                    self.embedder.descriptor().model,
                    entries.len(),
                    self.ranked.config().rrf_k,
                    match &self.reranker {
                        Some((r, n)) => {
                            let d = r.descriptor();
                            format!(";rerank={}:{}@{}(top={n})", d.provider, d.model, d.revision)
                        }
                        None => String::new(),
                    }
                )
            }
            None => "memory=none".into(),
        };
        let mut focus_method = String::new();
        if focus {
            // The asked participant's own turns as one more fused list.
            let asked = Grounding::of(&query.text, &people).map(|g| g.asked);
            let ranked: Vec<String> = order.iter().map(usize::to_string).collect();
            let own: Vec<String> = order
                .iter()
                .filter(|&&i| match (&units[i].kind, &asked) {
                    (UnitKind::Turn(t) | UnitKind::Part(t, _), Some(asked)) => turns
                        .speaker(t)
                        .is_some_and(|s| asked.iter().any(|a| a == s)),
                    _ => false,
                })
                .map(usize::to_string)
                .collect();
            focus_method = format!(
                ";focus=v8(query={},own={})",
                if focused.is_some() {
                    "unnamed"
                } else {
                    "as-asked"
                },
                own.len()
            );
            if !own.is_empty() {
                order = rrf(&[ranked, own], self.ranked.config().rrf_k)
                    .into_iter()
                    .map(|(i, _)| i.parse().expect("index"))
                    .collect();
            }
        }
        let mut window_method = String::new();
        if let Some(w) = window {
            // Re-fuse the final order with the units from inside the
            // question's window, so they rise without hiding the rest.
            let ranked: Vec<String> = order.iter().map(usize::to_string).collect();
            let inside: Vec<String> = order
                .iter()
                .filter(|&&i| match &units[i].kind {
                    UnitKind::Turn(t) | UnitKind::Part(t, _) => {
                        turns.day(t).is_some_and(|d| w.contains(d))
                    }
                    UnitKind::Fact => false,
                })
                .map(usize::to_string)
                .collect();
            window_method = format!(";window={}..{}({} units)", w.from, w.to, inside.len());
            order = rrf(&[ranked, inside], self.ranked.config().rrf_k)
                .into_iter()
                .map(|(i, _)| i.parse().expect("index"))
                .collect();
        }

        let fill = self.fill(&corpus, &units, &order, &turns, budget).await?;
        let tokens = self.tokenizer.count(&fill.text);
        let retrieved_ids = order.iter().map(|&i| units[i].id.clone()).collect();
        let dropped_turns = visibility.turns.saturating_sub(fill.turns.len());
        Ok(Built::Ready(ContextBundle {
            format: CONTEXT_FORMAT.into(),
            text: fill.text,
            tokens,
            budget_tokens: Some(budget),
            method: Some(format!(
                "keep-ranked;{search_method};{memory_method}{focus_method}{window_method};fill=greedy,dedup=turn,order=conversation;budget={budget}{}",
                if decompose {
                    ";decompose=facets-v1(max=3)"
                } else {
                    ""
                }
            )),
            retrieved_ids,
            included_ids: fill.ids,
            included_turn_ids: fill.turns.into_iter().collect(),
            visibility,
            truncation: Some(Truncation {
                reason: "ranked fill to the arm budget".into(),
                dropped_chunks: order.len() - fill.included,
                dropped_turns,
                dropped_tokens: 0,
            }),
            blocks: vec![BlockTokens::new(
                "keep-ranked",
                Slot::Evidence,
                tokens,
                budget,
            )],
        }))
    }

    /// K2's or Mk2's whole budget.
    fn ranked_budget(&self, path: KeepPath) -> usize {
        if path.uses_memory() {
            self.retrieval.standing_tokens + self.retrieval.evidence_tokens
        } else {
            self.retrieval.evidence_tokens
        }
    }

    /// K3 and Mk3: the dated facts block, then K2's or Mk2's ranked fill in
    /// what the block leaves of the same budget.
    async fn try_build_with_facts(
        &mut self,
        path: KeepPath,
        representation: Representation,
        memory: Option<&mut MemoryArm>,
        history: &History,
        query: &Query,
        opts: FactOptions,
    ) -> anyhow::Result<Built> {
        if opts.assistant_route
            && keep_knowledge::query_intent::asks_about_assistant_replies(&query.text)
        {
            // Extracted facts cover what the user said; a question about
            // the assistant's own earlier reply needs that reply verbatim,
            // so it gets the plain ranked fill over the whole budget.
            let built = self
                .try_build_ranked(
                    path,
                    representation,
                    memory,
                    history,
                    query,
                    None,
                    None,
                    opts.decompose,
                    opts.focus,
                )
                .await?;
            return Ok(match built {
                Built::Ready(mut b) => {
                    b.method = Some(format!(
                        "keep-facts;route=assistant-reply(ranked only);{}",
                        b.method.unwrap_or_default()
                    ));
                    Built::Ready(b)
                }
                other => other,
            });
        }
        let total = self.ranked_budget(path);
        let state = self
            .facts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("extracted facts are not enabled for this run"))?;
        let facts_tokens = state.facts_tokens.min(total);
        let profile_tokens = state.profile_tokens;
        let window = match (opts.temporal, query.asked_at) {
            (true, Some(asked)) => question_window(&query.text, asked.date_naive()),
            _ => None,
        };
        // The corpus build also collects the preferences the profile needs.
        self.facts_corpus(history).await?;
        let profile = if opts.preferences {
            self.preference_block(history, query, profile_tokens)
                .await?
        } else {
            crate::memory_arm::ProfileBlock::default()
        };
        let mut memory = memory;
        let mut prune: Option<FactPrune> = None;
        let mut first_facts = 0;
        let (facts, facts_method, summaries, summaries_method, summary_tokens, rest, mut b) = loop {
            let (facts, facts_method) = self
                .facts_block(
                    history,
                    query,
                    facts_tokens,
                    opts,
                    window.as_ref(),
                    prune.as_ref(),
                )
                .await?;
            let summary_tokens = if opts.summaries {
                let left = total.saturating_sub(profile.tokens + facts.tokens);
                self.facts
                    .as_ref()
                    .map_or(0, |s| s.summary_tokens)
                    .min(left / 2)
            } else {
                0
            };
            let (summaries, summaries_method) = if opts.summaries {
                let (b, m) = self
                    .summaries_block(history, query, summary_tokens, window.as_ref())
                    .await?;
                (b, format!("{m};"))
            } else {
                (crate::facts::SummariesBlock::default(), String::new())
            };
            let mut header = String::new();
            if !profile.text.is_empty() {
                header.push_str(&profile.text);
                header.push_str("\n\n");
            }
            if !facts.text.is_empty() {
                header.push_str(&facts.text);
                header.push_str("\n\n");
            }
            if !summaries.text.is_empty() {
                header.push_str(&summaries.text);
                header.push_str("\n\n");
            }
            if !header.is_empty() {
                header.push_str(crate::facts::EVIDENCE_HEADER);
                header.push('\n');
            }
            // Joining can merge tokens at the seam; keep a small margin.
            let rest = total.saturating_sub(self.tokenizer.count(&header) + 4);
            let built = self
                .try_build_ranked(
                    path,
                    representation,
                    memory.as_deref_mut(),
                    history,
                    query,
                    Some(rest),
                    window.as_ref(),
                    opts.decompose,
                    opts.focus,
                )
                .await?;
            let Built::Ready(mut b) = built else {
                return Ok(built);
            };
            if opts.verbatim_first && prune.is_none() && !facts.fact_ids.is_empty() {
                // Once more without the facts whose source turns the
                // excerpts already show whole, so the excerpts get the room.
                let turns = TurnIndex::new(history);
                let shown: BTreeSet<String> = b
                    .included_turn_ids
                    .iter()
                    .filter(|t| turns.whole(t).is_some_and(|w| b.text.contains(&w)))
                    .cloned()
                    .collect();
                if !shown.is_empty() {
                    first_facts = facts.fact_ids.len();
                    prune = Some(FactPrune {
                        shown,
                        only: facts.fact_ids.iter().cloned().collect(),
                    });
                    continue;
                }
            }
            b.text = format!("{header}{}", b.text);
            break (
                facts,
                facts_method,
                summaries,
                summaries_method,
                summary_tokens,
                rest,
                b,
            );
        };
        let ranked_tokens = b.tokens;
        b.tokens = self.tokenizer.count(&b.text);
        b.budget_tokens = Some(total);
        let profile_method = if opts.preferences {
            format!(
                "profile=memory({},{} lines,budget={profile_tokens});",
                crate::memory_arm::PREFERENCE_METHOD,
                profile.lines
            )
        } else {
            String::new()
        };
        let verbatim_method = match &prune {
            Some(_) => format!(
                "verbatim_first(pruned={});",
                first_facts.saturating_sub(facts.fact_ids.len())
            ),
            None => String::new(),
        };
        b.method = Some(format!(
            "keep-facts;{profile_method}{facts_method};{verbatim_method}{summaries_method}{}",
            b.method.unwrap_or_default()
        ));
        let mut turns: BTreeSet<String> = b.included_turn_ids.into_iter().collect();
        turns.extend(facts.turns.iter().cloned());
        turns.extend(profile.turns.iter().cloned());
        b.included_turn_ids = turns.into_iter().collect();
        let mut ids = facts.fact_ids.clone();
        ids.extend(b.included_ids);
        b.included_ids = ids;
        let head_tokens = b.tokens - ranked_tokens.min(b.tokens);
        b.blocks = Vec::new();
        if opts.preferences {
            b.blocks.push(BlockTokens::new(
                "keep-profile",
                Slot::Standing,
                profile.tokens,
                profile_tokens,
            ));
        }
        b.blocks.push(BlockTokens::new(
            "keep-facts",
            Slot::Evidence,
            head_tokens.saturating_sub(profile.tokens + summaries.tokens),
            facts_tokens,
        ));
        if opts.summaries {
            b.blocks.push(BlockTokens::new(
                "keep-summaries",
                Slot::Evidence,
                summaries.tokens,
                summary_tokens,
            ));
        }
        b.blocks.push(BlockTokens::new(
            "keep-ranked",
            Slot::Evidence,
            ranked_tokens,
            rest,
        ));
        anyhow::ensure!(
            b.tokens <= total,
            "facts context is {} tokens, over its {total}-token budget",
            b.tokens
        );
        Ok(Built::Ready(b))
    }

    /// Take units in rank order while they fit, skipping any whose source
    /// turns are already shown, then present them in conversation order.
    async fn fill(
        &mut self,
        corpus: &str,
        units: &[Unit],
        order: &[usize],
        turns: &TurnIndex<'_>,
        budget: usize,
    ) -> anyhow::Result<Fill> {
        let access = Access::to([PARTITION]);
        // Turns shown whole, and turns with some part shown.
        let mut whole: BTreeSet<String> = BTreeSet::new();
        let mut partial: BTreeSet<String> = BTreeSet::new();
        let mut texts: std::collections::HashSet<&str> = Default::default();
        let mut taken: Vec<Taken> = Vec::new();
        let mut used = 0;
        for (rank, &i) in order.iter().enumerate() {
            let u = &units[i];
            let cited = match &u.kind {
                UnitKind::Turn(t) => {
                    if whole.contains(t) || partial.contains(t) {
                        continue;
                    }
                    vec![t.clone()]
                }
                UnitKind::Part(t, _) => {
                    if whole.contains(t) {
                        continue;
                    }
                    vec![t.clone()]
                }
                UnitKind::Fact => {
                    let search = self.last.as_mut().expect("searched first");
                    if !search.cites.contains_key(&u.id) {
                        let cited = self
                            .ranked
                            .cite(corpus, &access, &u.id)
                            .await?
                            .into_iter()
                            .map(|c| c.episode_id)
                            .collect();
                        search.cites.insert(u.id.clone(), cited);
                    }
                    let cited = search.cites[&u.id].clone();
                    if !cited.is_empty()
                        && cited
                            .iter()
                            .all(|t| whole.contains(t) || partial.contains(t))
                    {
                        continue;
                    }
                    cited
                }
            };
            let choice = std::iter::once(u.text.as_str())
                .chain(u.fallback.as_deref())
                .find(|t| used + self.tokenizer.count(t) < budget);
            let Some(text) = choice else { continue };
            if !texts.insert(text) {
                continue;
            }
            used += self.tokenizer.count(text) + 1;
            match &u.kind {
                UnitKind::Turn(t) => {
                    whole.insert(t.clone());
                }
                UnitKind::Part(t, _) => {
                    partial.insert(t.clone());
                }
                UnitKind::Fact => {}
            }
            let pos = match &u.kind {
                UnitKind::Turn(t) => (turns.position(t), 0, rank),
                UnitKind::Part(t, start) => (turns.position(t), *start, rank),
                UnitKind::Fact => (
                    cited
                        .iter()
                        .map(|t| turns.position(t))
                        .min()
                        .unwrap_or(usize::MAX),
                    0,
                    rank,
                ),
            };
            taken.push(Taken {
                pos,
                text,
                id: &u.id,
                cited,
            });
        }
        // Joined token counts can exceed the sum of the parts; drop the
        // lowest-ranked units until the whole context fits.
        loop {
            let mut shown: Vec<&Taken> = taken.iter().collect();
            shown.sort_by_key(|t| t.pos);
            let text = shown.iter().map(|t| t.text).collect::<Vec<_>>().join("\n");
            if self.tokenizer.count(&text) <= budget || taken.is_empty() {
                return Ok(Fill {
                    text,
                    ids: shown.iter().map(|t| t.id.to_owned()).collect(),
                    turns: shown.iter().flat_map(|t| t.cited.iter().cloned()).collect(),
                    included: shown.len(),
                });
            }
            taken.pop();
        }
    }

    /// Remove every corpus this run created. Returns warnings, never fails.
    pub async fn cleanup(&mut self) -> Vec<String> {
        let mut warnings = Vec::new();
        let facts = self
            .facts
            .as_mut()
            .map(|f| f.corpora.drain().collect::<Vec<_>>())
            .unwrap_or_default();
        for (_, corpus) in self.corpora.drain().chain(facts) {
            if let Err(e) = self.k.delete_corpus(&corpus).await {
                warnings.push(format!("could not delete knowledge corpus {corpus}: {e}"));
            }
        }
        warnings
    }
}

/// What `verbatim_first` leaves out of the facts block on its second pass.
struct FactPrune {
    /// Turns the first pass's excerpts showed whole.
    shown: BTreeSet<String>,
    /// The facts the first pass showed; no other fact takes the room.
    only: BTreeSet<String>,
}

impl FactPrune {
    /// A fact the first pass showed stays when it adds to the excerpts:
    /// its attribute `changed` value over time, some turn it cites is not
    /// shown whole, or it carries its age at the question date, which no
    /// excerpt shows.
    fn keeps(&self, line: &crate::facts::FactLine, changed: bool) -> bool {
        self.only.contains(&line.id)
            && (changed
                || line.age_days.is_some()
                || line.turns.is_empty()
                || line.turns.iter().any(|t| !self.shown.contains(t)))
    }
}

/// One corpus per history: a source per session and an episode per turn,
/// ingested in date order with `clock` at the session date.
async fn ingest_history(
    k: &Knowledge,
    clock: &ManualClock,
    corpus: &str,
    history: &History,
) -> anyhow::Result<()> {
    let _t = timing::span("bench.knowledge_corpus").items(1);
    // Leftovers from an interrupted run with the same manifest.
    k.delete_corpus(corpus).await?;
    let mut sessions: Vec<_> = history.sessions.iter().collect();
    sessions.sort_by_key(|s| s.date.unwrap_or(chrono::DateTime::UNIX_EPOCH));
    for s in sessions {
        let at = s.date.unwrap_or(chrono::DateTime::UNIX_EPOCH);
        clock.set(at);
        let episodes: Vec<EpisodeInput> = s
            .turns
            .iter()
            .filter(|t| !t.text.trim().is_empty())
            .map(|t| EpisodeInput {
                id: t.id.clone(),
                speaker: t.role.label().into(),
                occurred_at: s.date,
                text: t.text.clone(),
            })
            .collect();
        k.ingest(
            corpus,
            SourceInput {
                id: s.id.clone(),
                partition: PARTITION.into(),
                collection: COLLECTION.into(),
                title: format!(
                    "Session {} ({})",
                    s.id,
                    s.date_raw.as_deref().unwrap_or("date unknown")
                ),
                kind: SourceKind::Conversation,
            },
            episodes,
        )
        .await?;
        // Everything derived from a session is recorded at its date.
        let t = timing::span("knowledge.run_jobs").items(1);
        let run = k.run_jobs(corpus, 50).await?;
        drop(t);
        if run.failed > 0 {
            anyhow::bail!("{} knowledge jobs failed for session {}", run.failed, s.id);
        }
    }
    Ok(())
}

/// A unit the fill kept, in the order it was taken.
struct Taken<'a> {
    /// Conversation position, then character offset, then rank.
    pos: (usize, i32, usize),
    text: &'a str,
    id: &'a str,
    cited: Vec<String>,
}

struct Fill {
    text: String,
    ids: Vec<String>,
    turns: BTreeSet<String>,
    included: usize,
}

/// Where each turn sits in the conversation and how to show it whole.
struct TurnIndex<'h> {
    turns: HashMap<&'h str, (usize, &'h crate::dataset::Turn, String)>,
    days: HashMap<&'h str, chrono::NaiveDate>,
    /// Each turn as the dense baselines embed it, so vectors are shared.
    sources: HashMap<&'h str, String>,
}

impl<'h> TurnIndex<'h> {
    /// Sessions in date order, as they were ingested.
    fn new(history: &'h History) -> Self {
        let mut sessions: Vec<_> = history.sessions.iter().collect();
        sessions.sort_by_key(|s| s.date.unwrap_or(chrono::DateTime::UNIX_EPOCH));
        let mut turns = HashMap::new();
        let mut sources = HashMap::new();
        let mut days = HashMap::new();
        for s in sessions {
            let date = s
                .date
                .map_or_else(|| "undated".into(), |d| d.date_naive().to_string());
            let raw = s.date_raw.as_deref().unwrap_or("date unknown");
            for t in &s.turns {
                let pos = turns.len();
                turns.insert(t.id.as_str(), (pos, t, date.clone()));
                if let Some(d) = s.date {
                    days.insert(t.id.as_str(), d.date_naive());
                }
                sources.insert(t.id.as_str(), format!("[{raw}] {}", render_turn(t)));
            }
        }
        Self {
            turns,
            days,
            sources,
        }
    }

    /// The date of the session the turn belongs to.
    fn day(&self, turn_id: &str) -> Option<chrono::NaiveDate> {
        self.days.get(turn_id).copied()
    }

    fn source(&self, turn_id: &str) -> Option<&str> {
        self.sources.get(turn_id).map(String::as_str)
    }

    fn position(&self, turn_id: &str) -> usize {
        self.turns.get(turn_id).map_or(usize::MAX, |t| t.0)
    }

    /// Who said the turn.
    fn speaker(&self, turn_id: &str) -> Option<&str> {
        self.turns.get(turn_id).map(|(_, t, _)| t.role.label())
    }

    /// The whole turn, rendered exactly as Knowledge renders a chunk.
    fn whole(&self, turn_id: &str) -> Option<String> {
        self.turns
            .get(turn_id)
            .map(|(_, t, date)| format!("[{date}] {}: {}", t.role.label(), t.text.trim()))
    }

    /// A Memory line, dated like the rest of the context.
    fn excerpt(&self, turn_id: &str, line: &str) -> String {
        match self.turns.get(turn_id) {
            Some((_, _, date)) => format!("[{date}] {line}"),
            None => line.to_owned(),
        }
    }

    fn is_whole(&self, turn_id: &str, quote: &str) -> bool {
        self.turns
            .get(turn_id)
            .is_some_and(|(_, t, _)| t.text.trim() == quote)
    }
}

fn knowledge_unit(h: &Hit, turns: &TurnIndex) -> Unit {
    let id = h.item.point_id();
    match &h.item {
        ItemRef::Chunk(_) => {
            let turn = h.citation.episode_id.clone();
            if turns.is_whole(&turn, &h.citation.quote) {
                Unit {
                    key: format!("turn:{turn}"),
                    id,
                    text: h.text.clone(),
                    fallback: None,
                    kind: UnitKind::Turn(turn),
                }
            } else {
                Unit {
                    key: id.clone(),
                    id,
                    text: h.text.clone(),
                    fallback: None,
                    kind: UnitKind::Part(turn, h.citation.char_start),
                }
            }
        }
        ItemRef::Fact(_) => Unit {
            key: id.clone(),
            id,
            text: h.text.clone(),
            fallback: None,
            kind: UnitKind::Fact,
        },
    }
}

/// A Memory entry Knowledge did not rank: its source turn, or the entry
/// itself when the turn does not fit.
/// The named speakers of `history`, in order of first appearance. Users
/// and assistants are not named, so they give no subject grounding.
fn participants(history: &History) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in history.sessions.iter().flat_map(|s| &s.turns) {
        if let keep_core::knowledge::Role::Speaker(name) = &t.role
            && !out.contains(name)
        {
            out.push(name.clone());
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn memory_unit(turn_id: &str, line: &str, turns: &TurnIndex) -> Unit {
    let excerpt = turns.excerpt(turn_id, line);
    let (text, fallback) = match turns.whole(turn_id) {
        Some(whole) => (whole, Some(excerpt)),
        None => (excerpt, None),
    };
    Unit {
        id: format!("turn:{turn_id}"),
        key: format!("turn:{turn_id}"),
        text,
        fallback,
        kind: UnitKind::Turn(turn_id.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::FactPrune;
    use crate::facts::FactLine;
    use keep_knowledge::model::{FactStatus, TimeBasis};

    fn fact(id: &str, turns: &[&str], age_days: Option<i64>) -> FactLine {
        FactLine {
            id: id.into(),
            subject: "Mira".into(),
            predicate: "started".into(),
            value: "a solar kiln".into(),
            text: "Mira started a solar kiln".into(),
            valid_from: None,
            valid_to: None,
            basis: TimeBasis::Observed,
            status: FactStatus::Active,
            turns: turns.iter().map(|t| (*t).into()).collect(),
            age_days,
        }
    }

    #[test]
    fn verbatim_first_drops_only_facts_the_excerpts_already_show() {
        let prune = FactPrune {
            shown: ["D1:1".into(), "D1:2".into()].into(),
            only: ["f1".into(), "f2".into(), "f3".into(), "f4".into()].into(),
        };
        // Every cited turn shown whole: the excerpt says it already.
        assert!(!prune.keeps(&fact("f1", &["D1:1", "D1:2"], None), false));
        // One cited turn not shown: the fact adds something.
        assert!(prune.keeps(&fact("f2", &["D1:1", "D2:1"], None), false));
        // No citation, an age at the question date, or a changed value:
        // nothing in the excerpts replaces it.
        assert!(prune.keeps(&fact("f3", &[], None), false));
        assert!(prune.keeps(&fact("f4", &["D1:1"], Some(3)), false));
        assert!(prune.keeps(&fact("f1", &["D1:1"], None), true));
        // A fact the first pass did not show never takes the freed room.
        assert!(!prune.keeps(&fact("f9", &["D9:9"], None), false));
    }
}
