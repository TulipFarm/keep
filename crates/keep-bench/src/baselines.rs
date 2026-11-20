//! Outside baselines.
//!
//! Each baseline turns the visible history and the question text into a
//! [`ContextBundle`]. None of them sees gold answers; only the explicitly
//! diagnostic oracle arm receives the gold evidence ids.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use keep_providers::{Embedder, Tokenizer, Usage, timing};
use keep_store::{Bm25Index, Bm25Params, LexicalIndex, VectorIndex, VectorPoint, lexical};

use crate::ablation::{ArmKind, Chunking};
use crate::context::{
    CONTEXT_FORMAT, Chunk, ContextBundle, Truncation, chunks, pack_ranked, render_full_history,
};
use crate::dataset::{History, Query};
use crate::manifest::{ModelSpec, RetrievalSpec};
use crate::prompt;

/// The result of building one context.
#[derive(Clone, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "built once per question and consumed immediately"
)]
pub enum Built {
    Ready(ContextBundle),
    /// The arm cannot serve this question by design (for example the full
    /// history exceeds the model's context). Reported, never hidden.
    Unavailable(String),
    /// Something broke while building context (an embedding or storage
    /// error). Counts as a failure.
    Failed(String),
}

/// Gold evidence handed only to the oracle diagnostic.
pub struct OracleEvidence<'a> {
    pub turn_ids: &'a [String],
    pub session_ids: &'a [String],
}

pub struct DenseBackend {
    pub embedder: Arc<dyn Embedder>,
    pub index: Arc<dyn VectorIndex>,
    pub collection: String,
    ingested: HashSet<String>,
    /// Embedding usage so far, for cost accounting.
    pub usage: Usage,
    pub batch_size: usize,
}

impl DenseBackend {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        index: Arc<dyn VectorIndex>,
        collection: String,
        batch_size: usize,
    ) -> Self {
        Self {
            embedder,
            index,
            collection,
            ingested: HashSet::new(),
            usage: Usage::default(),
            batch_size: batch_size.max(1),
        }
    }

    pub fn ingested_namespaces(&self) -> impl Iterator<Item = &String> {
        self.ingested.iter()
    }

    fn add_usage(&mut self, u: Usage) {
        self.usage.input_tokens += u.input_tokens;
        self.usage.output_tokens += u.output_tokens;
        self.usage.provider_reported = u.provider_reported;
    }

    async fn ensure_history(
        &mut self,
        history: &History,
        chunking: Chunking,
    ) -> Result<(), String> {
        if self.ingested.contains(&history.id) {
            return Ok(());
        }
        self.index
            .ensure_collection(&self.collection, self.embedder.dimensions())
            .await
            .map_err(|e| e.to_string())?;
        let usage = self.writer().write(history, chunking).await?;
        self.add_usage(usage);
        self.ingested.insert(history.id.clone());
        Ok(())
    }

    fn writer(&self) -> HistoryWriter {
        HistoryWriter {
            embedder: self.embedder.clone(),
            index: self.index.clone(),
            collection: self.collection.clone(),
            batch_size: self.batch_size,
        }
    }

    /// Embed and store every history not stored yet, up to `concurrency` at
    /// a time. Histories have their own namespaces and searches are exact,
    /// so the result is what one-at-a-time ingestion leaves. A history that
    /// fails here is left for the lazy path to retry and report.
    pub async fn prepare(
        &mut self,
        histories: &[&History],
        chunking: Chunking,
        concurrency: usize,
    ) -> Vec<String> {
        if let Err(e) = self
            .index
            .ensure_collection(&self.collection, self.embedder.dimensions())
            .await
        {
            return vec![format!("dense collection: {e}")];
        }
        let gate = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
        let mut tasks = tokio::task::JoinSet::new();
        let mut queued = HashSet::new();
        for h in histories {
            if self.ingested.contains(&h.id) || !queued.insert(h.id.clone()) {
                continue;
            }
            let (writer, history, gate) = (self.writer(), (*h).clone(), gate.clone());
            tasks.spawn(async move {
                let _permit = gate.acquire_owned().await.expect("semaphore open");
                let r = writer.write(&history, chunking).await;
                (history.id, r)
            });
        }
        let mut errors = Vec::new();
        while let Some(done) = tasks.join_next().await {
            match done {
                Ok((id, Ok(usage))) => {
                    self.add_usage(usage);
                    self.ingested.insert(id);
                }
                Ok((id, Err(e))) => errors.push(format!("dense ingest of {id}: {e}")),
                Err(e) => errors.push(format!("dense ingest task: {e}")),
            }
        }
        errors
    }

    /// Remove every namespace this backend created.
    pub async fn cleanup(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        for ns in self.ingested.drain() {
            if let Err(e) = self.index.delete_namespace(&self.collection, &ns).await {
                errors.push(format!("cleanup of {ns}: {e}"));
            }
        }
        errors
    }
}

/// Embeds one history's chunks and stores them in its namespace.
struct HistoryWriter {
    embedder: Arc<dyn Embedder>,
    index: Arc<dyn VectorIndex>,
    collection: String,
    batch_size: usize,
}

impl HistoryWriter {
    async fn write(&self, history: &History, chunking: Chunking) -> Result<Usage, String> {
        // Start clean in case an earlier run left points behind.
        self.index
            .delete_namespace(&self.collection, &history.id)
            .await
            .map_err(|e| e.to_string())?;
        let t = timing::span("dense.chunk");
        let all = chunks(history, None, chunking);
        drop(t.items(all.len()));
        let mut usage = Usage::default();
        let mut points = Vec::with_capacity(all.len());
        for batch in all.chunks(self.batch_size) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let t = timing::span("dense.embed").items(texts.len());
            let resp = self
                .embedder
                .embed(&texts)
                .await
                .map_err(|e| e.to_string())?;
            drop(t);
            usage.input_tokens += resp.usage.input_tokens;
            usage.output_tokens += resp.usage.output_tokens;
            usage.provider_reported = resp.usage.provider_reported;
            points.extend(
                batch
                    .iter()
                    .zip(resp.vectors)
                    .map(|(c, vector)| VectorPoint {
                        id: c.id.clone(),
                        vector,
                    }),
            );
        }
        let _t = timing::span("dense.vector_upsert").items(points.len());
        self.index
            .upsert(&self.collection, &history.id, &points)
            .await
            .map_err(|e| e.to_string())?;
        Ok(usage)
    }
}

pub struct Baselines {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub answer_model: ModelSpec,
    pub retrieval: RetrievalSpec,
    pub dense: Option<DenseBackend>,
    bm25: HashMap<(String, usize), (Bm25Index, Vec<Chunk>)>,
    pub(crate) memory: crate::memory_arm::MemoryArm,
}

const GREP_STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "which", "who", "whom",
    "why", "how", "did", "does", "has", "have", "had", "you", "your", "his", "her", "their",
    "they", "them", "she", "that", "this", "with", "from", "about", "into", "any", "not", "but",
    "can", "could", "would", "should", "will",
];

pub const NOTES_GREP_METHOD: &str =
    "notes-grep-v1;terms=lower-alnum,len>=3;stopwords=fixed-v1;rank=distinct-terms,recency";

impl Baselines {
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        answer_model: ModelSpec,
        retrieval: RetrievalSpec,
        dense: Option<DenseBackend>,
    ) -> Self {
        Self {
            tokenizer,
            answer_model,
            retrieval,
            dense,
            bm25: HashMap::new(),
            memory: Default::default(),
        }
    }

    fn bundle(&self, text: String) -> ContextBundle {
        ContextBundle {
            format: CONTEXT_FORMAT.into(),
            tokens: self.tokenizer.count(&text),
            text,
            ..Default::default()
        }
    }

    pub async fn build(
        &mut self,
        arm: &ArmKind,
        history: &History,
        query: &Query,
        oracle: Option<OracleEvidence<'_>>,
    ) -> Built {
        if let Some(reason) = arm.unavailable_reason() {
            return Built::Unavailable(reason);
        }
        let tok = self.tokenizer.clone();
        let (_, visibility) = render_full_history(history, query.asked_at);
        let mut built = match arm {
            ArmKind::NoMemory => Built::Ready(self.bundle(String::new())),
            ArmKind::FullContext => self.full_context(history, query),
            ArmKind::RecentWindow { budget_tokens } => {
                let all = chunks(history, query.asked_at, Chunking::Turn);
                let mut used = 0;
                let mut take = 0;
                for c in all.iter().rev() {
                    let cost = tok.count(&c.text) + 1;
                    if used + cost > *budget_tokens {
                        break;
                    }
                    used += cost;
                    take += 1;
                }
                let ranked: Vec<&Chunk> = all.iter().rev().take(take).collect();
                let (text, ids, turns, tokens, trunc) = pack_ranked(
                    &ranked,
                    &all,
                    *budget_tokens,
                    tok.as_ref(),
                    "recent-window budget",
                );
                Built::Ready(ContextBundle {
                    format: CONTEXT_FORMAT.into(),
                    text,
                    tokens,
                    budget_tokens: Some(*budget_tokens),
                    method: Some("most-recent-turns".into()),
                    retrieved_ids: ids.clone(),
                    included_ids: ids,
                    included_turn_ids: turns,
                    visibility,
                    truncation: (trunc.dropped_chunks > 0).then_some(trunc),
                    blocks: Vec::new(),
                })
            }
            ArmKind::Bm25 {
                budget_tokens,
                top_k,
            } => {
                let key = (history.id.clone(), visibility.sessions);
                let chunking = self.retrieval.chunking;
                let params = Bm25Params {
                    k1: self.retrieval.bm25_k1,
                    b: self.retrieval.bm25_b,
                };
                let (index, all) = self.bm25.entry(key).or_insert_with(|| {
                    let all = chunks(history, query.asked_at, chunking);
                    let index = Bm25Index::build(
                        params,
                        all.iter().map(|c| (c.id.clone(), c.text.clone())),
                    );
                    (index, all)
                });
                let hits = index.search(&query.text, *top_k);
                let by_id: HashMap<&str, &Chunk> = all.iter().map(|c| (c.id.as_str(), c)).collect();
                let ranked: Vec<&Chunk> = hits
                    .iter()
                    .filter_map(|h| by_id.get(h.id.as_str()).copied())
                    .collect();
                let (text, ids, turns, tokens, trunc) = pack_ranked(
                    &ranked,
                    all,
                    *budget_tokens,
                    tok.as_ref(),
                    "retrieval top-k and budget",
                );
                Built::Ready(ContextBundle {
                    format: CONTEXT_FORMAT.into(),
                    text,
                    tokens,
                    budget_tokens: Some(*budget_tokens),
                    method: Some(format!(
                        "{};chunking={chunking:?};top_k={top_k}",
                        index.method()
                    )),
                    retrieved_ids: hits.into_iter().map(|h| h.id).collect(),
                    included_ids: ids,
                    included_turn_ids: turns,
                    visibility,
                    truncation: Some(trunc),
                    blocks: Vec::new(),
                })
            }
            ArmKind::Dense {
                budget_tokens,
                top_k,
            } => self.dense(history, query, *budget_tokens, *top_k).await,
            ArmKind::NotesGrep { budget_tokens } => {
                let all = chunks(history, query.asked_at, Chunking::Turn);
                let terms: HashSet<String> = lexical::tokenize(&query.text)
                    .into_iter()
                    .filter(|t| t.chars().count() >= 3 && !GREP_STOPWORDS.contains(&t.as_str()))
                    .collect();
                let mut scored: Vec<(usize, &Chunk)> = all
                    .iter()
                    .filter_map(|c| {
                        let words: HashSet<String> =
                            lexical::tokenize(&c.text).into_iter().collect();
                        let n = terms.iter().filter(|t| words.contains(*t)).count();
                        (n > 0).then_some((n, c))
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.order.cmp(&a.1.order)));
                let ranked: Vec<&Chunk> = scored.iter().map(|(_, c)| *c).collect();
                let (text, ids, turns, tokens, trunc) = pack_ranked(
                    &ranked,
                    &all,
                    *budget_tokens,
                    tok.as_ref(),
                    "grep matches and budget",
                );
                Built::Ready(ContextBundle {
                    format: CONTEXT_FORMAT.into(),
                    text,
                    tokens,
                    budget_tokens: Some(*budget_tokens),
                    method: Some(NOTES_GREP_METHOD.into()),
                    retrieved_ids: ranked.iter().map(|c| c.id.clone()).collect(),
                    included_ids: ids,
                    included_turn_ids: turns,
                    visibility,
                    truncation: Some(trunc),
                    blocks: Vec::new(),
                })
            }
            ArmKind::OracleEvidence => match oracle {
                None => Built::Unavailable("oracle evidence was not provided".into()),
                Some(ev) => {
                    let all = chunks(history, query.asked_at, Chunking::Turn);
                    let turn_set: HashSet<&str> = ev.turn_ids.iter().map(String::as_str).collect();
                    let session_set: HashSet<&str> =
                        ev.session_ids.iter().map(String::as_str).collect();
                    let pick: Vec<&Chunk> = if turn_set.is_empty() {
                        all.iter()
                            .filter(|c| session_set.contains(c.session_id.as_str()))
                            .collect()
                    } else {
                        all.iter()
                            .filter(|c| turn_set.contains(c.id.as_str()))
                            .collect()
                    };
                    if pick.is_empty() {
                        Built::Unavailable("question has no visible gold evidence".into())
                    } else {
                        let text = pick
                            .iter()
                            .map(|c| c.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        let ids: Vec<String> = pick.iter().map(|c| c.id.clone()).collect();
                        Built::Ready(ContextBundle {
                            format: CONTEXT_FORMAT.into(),
                            tokens: tok.count(&text),
                            text,
                            budget_tokens: None,
                            method: Some("gold-evidence-diagnostic".into()),
                            retrieved_ids: ids.clone(),
                            included_turn_ids: ids.clone(),
                            included_ids: ids,
                            visibility,
                            truncation: None,
                            blocks: Vec::new(),
                        })
                    }
                }
            },
            ArmKind::Keep {
                path: crate::ablation::KeepPath::M,
                representation,
            } => {
                self.memory
                    .build(
                        *representation,
                        self.retrieval.standing_tokens,
                        history,
                        query,
                        tok.as_ref(),
                    )
                    .await
            }
            ArmKind::FilesystemAgent { .. } => unreachable!("handled above"),
            ArmKind::Keep { .. } => Built::Failed("Keep arms are built by the runner".into()),
        };
        if let Built::Ready(b) = &mut built {
            b.visibility = visibility;
        }
        built
    }

    /// Prompt tokens for a rendered context, using the same template as the
    /// answer call.
    pub fn prompt_tokens(&self, context: &str, query: &Query) -> usize {
        prompt::answer_messages(context, query)
            .iter()
            .map(|m| self.tokenizer.count(&m.content))
            .sum()
    }

    fn full_context(&self, history: &History, query: &Query) -> Built {
        let (text, visibility) = render_full_history(history, query.asked_at);
        let needed =
            self.prompt_tokens(&text, query) + self.answer_model.max_output_tokens as usize;
        if needed > self.answer_model.context_limit {
            return Built::Unavailable(format!(
                "unavailable due to context capacity: needs {needed} tokens including the output reservation, \
                 limit is {} ({})",
                self.answer_model.context_limit,
                self.tokenizer.id()
            ));
        }
        let all = chunks(history, query.asked_at, Chunking::Turn);
        let ids: Vec<String> = all.iter().map(|c| c.id.clone()).collect();
        Built::Ready(ContextBundle {
            format: CONTEXT_FORMAT.into(),
            tokens: self.tokenizer.count(&text),
            text,
            budget_tokens: None,
            method: Some("complete-visible-history".into()),
            retrieved_ids: Vec::new(),
            included_turn_ids: ids.clone(),
            included_ids: ids,
            visibility,
            truncation: None,
            blocks: Vec::new(),
        })
    }

    async fn dense(
        &mut self,
        history: &History,
        query: &Query,
        budget: usize,
        top_k: usize,
    ) -> Built {
        let chunking = self.retrieval.chunking;
        let tok = self.tokenizer.clone();
        let Some(dense) = self.dense.as_mut() else {
            return Built::Failed(
                "dense arm configured without an embedding/vector backend".into(),
            );
        };
        if let Err(e) = dense.ensure_history(history, chunking).await {
            return Built::Failed(format!("dense ingestion failed: {e}"));
        }
        let visible = chunks(history, query.asked_at, chunking);
        let hidden = chunks(history, None, chunking).len() - visible.len();
        let t = timing::span("dense.query_embed").items(1);
        let qv = match dense
            .embedder
            .embed(std::slice::from_ref(&query.text))
            .await
        {
            Ok(r) => {
                dense.add_usage(r.usage);
                r.vectors.into_iter().next().unwrap_or_default()
            }
            Err(e) => return Built::Failed(format!("query embedding failed: {e}")),
        };
        drop(t);
        let t = timing::span("dense.search").items(1);
        let hits = match dense
            .index
            .search(&dense.collection, &history.id, &qv, top_k + hidden)
            .await
        {
            Ok(h) => h,
            Err(e) => return Built::Failed(format!("vector search failed: {e}")),
        };
        drop(t);
        let by_id: HashMap<&str, &Chunk> = visible.iter().map(|c| (c.id.as_str(), c)).collect();
        let ranked: Vec<&Chunk> = hits
            .iter()
            .filter_map(|h| by_id.get(h.id.as_str()).copied())
            .take(top_k)
            .collect();
        let (text, ids, turns, tokens, trunc) = pack_ranked(
            &ranked,
            &visible,
            budget,
            tok.as_ref(),
            "retrieval top-k and budget",
        );
        let caps = dense.index.capabilities();
        let desc = dense.embedder.descriptor();
        Built::Ready(ContextBundle {
            format: CONTEXT_FORMAT.into(),
            text,
            tokens,
            budget_tokens: Some(budget),
            method: Some(format!(
                "dense;embedder={}@{};backend={};exact={};distance={:?};chunking={chunking:?};top_k={top_k}",
                desc.model, desc.revision, caps.backend, caps.exact_search, caps.distance
            )),
            retrieved_ids: ranked.iter().map(|c| c.id.clone()).collect(),
            included_ids: ids,
            included_turn_ids: turns,
            visibility: Default::default(),
            truncation: Some(trunc),
            blocks: Vec::new(),
        })
    }
}

/// A truncation record for the report: `None` means nothing visible was cut.
pub fn truncation_of(b: &ContextBundle) -> Option<&Truncation> {
    b.truncation.as_ref().filter(|t| t.dropped_chunks > 0)
}
