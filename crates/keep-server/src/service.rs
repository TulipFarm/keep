//! [`KeepService`]: the one service both transports call.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, anyhow, bail};
use chrono::{DateTime, NaiveDate, Utc};
use keep_knowledge::engine::JobRetention;
use keep_knowledge::model::JobCounts;
use keep_knowledge::session_extract::{SessionExtractor, SessionInput, SessionTurn, text_turns};
use keep_knowledge::session_summary::SessionSummarizer;
use keep_knowledge::{
    Access, AsOf, DEFAULT_EVIDENCE_TOKENS, EpisodeInput, EvidenceAllowance, IndexRequest,
    Knowledge, KnowledgeConfig, MAX_EVIDENCE_TOKENS, SearchRequest, SourceInput, SourceKind,
    profile,
};
use keep_memory::context::{DEFAULT_CONTEXT_TOKENS, LABEL_KNOWLEDGE, MAX_CONTEXT_TOKENS};
use keep_memory::portable::{InstructionPolicy, export_bundle};
use keep_memory::service::FileView;
use keep_memory::{
    ContextBudget, ContextRequest, CreateFile, Edit, ForgetSelector, Head, InMemoryJournal,
    InMemoryLedger, KnowledgeToc, ListQuery, MemoryError, MemoryService, MetadataPatch, OpCtx,
};
use keep_providers::{ProviderError, Tokenizer, tokenizer_for};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::api::*;
use crate::budget::{BudgetConfig, BudgetPer, EvidenceBudget, Reservation};
use crate::config::{ServiceArgs, StorageProfile};
use crate::embedder;
use crate::error::{ApiError, ErrorCode};
use crate::model::ModelSetup;
use crate::scope::{Scope, ScopeArg};
use crate::settings::{Settings, check_embedder, database_receipt_key, file_receipt_key};

/// Rounds of jobs a projection rebuild runs on top of one per 64 queued.
const REBUILD_EXTRA_ROUNDS: usize = 50;
/// Default time an ingest spends on its own source's jobs before answering.
pub const DEFAULT_INGEST_JOB_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);
const MAX_LIST_PAGE: usize = 100;
const DEFAULT_COLLECTION: &str = "general";
/// Bounds on what one `knowledge_summarize` call reads.
const MAX_SUMMARIZE_EPISODES: usize = 200;
const MAX_SUMMARIZE_CHARS: usize = 200_000;
/// Scopes and job batches the background worker takes per tick.
const WORKER_CORPORA: usize = 20;
const WORKER_ROUNDS: usize = 10;
/// How often `keep serve` looks for due background jobs.
pub const WORKER_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

pub struct ServiceOptions {
    pub defaults: Scope,
    /// Recorded as the harness on Memory revisions.
    pub agent: Option<String>,
    pub tokenizer: Arc<dyn Tokenizer>,
    /// Whether callers may write the always-included profile.
    pub profile_writes: bool,
    /// Key that signs idempotency receipts. Random per process if unset.
    pub receipt_key: Option<Vec<u8>>,
    /// Write one metadata-only line per call to stderr.
    pub log_calls: bool,
    /// Cap on Knowledge evidence per caller over a rolling window.
    pub evidence_budget: Option<BudgetConfig>,
    /// Chat model for fact extraction and `knowledge_summarize`.
    pub model: Option<ModelSetup>,
    /// Time an ingest spends running its own source's jobs before it
    /// answers; the background worker does the rest. Zero runs none inline.
    pub ingest_job_budget: std::time::Duration,
    /// Pause between the background worker's job batches.
    pub job_pause: std::time::Duration,
    /// How long finished background jobs are kept.
    pub job_retention: JobRetention,
}

impl ServiceOptions {
    pub fn new(defaults: Scope) -> Self {
        Self {
            defaults,
            agent: None,
            tokenizer: Arc::new(keep_providers::ApproxTokenizer),
            profile_writes: true,
            receipt_key: None,
            log_calls: false,
            evidence_budget: None,
            model: None,
            ingest_job_budget: DEFAULT_INGEST_JOB_BUDGET,
            job_pause: std::time::Duration::ZERO,
            job_retention: JobRetention::default(),
        }
    }
}

pub struct KeepService {
    memory: MemoryService,
    knowledge: Knowledge,
    tokenizer: Arc<dyn Tokenizer>,
    defaults: Scope,
    agent: Option<String>,
    profile_writes: bool,
    log_calls: bool,
    storage: &'static str,
    budget: Option<EvidenceBudget>,
    model: Option<ModelSetup>,
    summarizer: Option<SessionSummarizer>,
    ingest_job_budget: std::time::Duration,
}

impl KeepService {
    pub fn new(
        memory: MemoryService,
        knowledge: Knowledge,
        storage: &'static str,
        opts: ServiceOptions,
    ) -> Self {
        let memory = match opts.receipt_key {
            Some(k) => memory.with_receipt_key(k),
            None => memory,
        };
        let mut knowledge = knowledge
            .with_tokenizer(opts.tokenizer.clone())
            .with_job_pause(opts.job_pause)
            .with_job_retention(opts.job_retention);
        let mut summarizer = None;
        if let Some(m) = &opts.model {
            if m.extraction {
                let sx =
                    SessionExtractor::new(Some(m.chat.clone()), m.id.clone(), None, m.reply_tokens);
                knowledge = knowledge.with_session_extractor(Arc::new(sx), m.concurrency);
            }
            summarizer = Some(
                SessionSummarizer::new(Some(m.chat.clone()), m.id.clone(), None, m.reply_tokens)
                    .for_documents(),
            );
        }
        Self {
            memory,
            knowledge,
            tokenizer: opts.tokenizer,
            defaults: opts.defaults,
            agent: opts.agent,
            profile_writes: opts.profile_writes,
            log_calls: opts.log_calls,
            storage,
            budget: opts.evidence_budget.map(EvidenceBudget::new),
            model: opts.model,
            summarizer,
            ingest_job_budget: opts.ingest_job_budget,
        }
    }

    /// Everything in this process, with the offline hash embedder.
    pub fn in_memory(opts: ServiceOptions) -> Self {
        let memory = MemoryService::new(
            Arc::new(InMemoryLedger::new()),
            Arc::new(InMemoryJournal::new()),
        );
        Self::new(
            memory,
            profile::in_memory(),
            StorageProfile::InMemory.as_str(),
            opts,
        )
    }

    /// Build from command-line flags and the environment.
    pub async fn open(args: &ServiceArgs) -> anyhow::Result<Self> {
        let tokenizer = tokenizer_for(&args.tokenizer)
            .ok_or_else(|| anyhow!("unknown tokenizer {:?}", args.tokenizer))?;
        let defaults = Scope::new(args.user.clone(), args.project.clone())
            .map_err(|e| anyhow!("default scope: {}", e.message))?;
        let mut opts = ServiceOptions {
            defaults,
            agent: args.agent.clone(),
            tokenizer,
            profile_writes: !args.no_profile_writes,
            receipt_key: std::env::var("KEEP_RECEIPT_KEY")
                .ok()
                .filter(|k| !k.is_empty())
                .map(String::into_bytes),
            log_calls: true,
            evidence_budget: args.evidence_budget()?,
            model: args.model.build()?,
            ingest_job_budget: std::time::Duration::from_millis(args.ingest_job_budget_ms),
            job_pause: std::time::Duration::from_millis(args.job_pause_ms),
            job_retention: args.job_retention(),
        };
        let config = KnowledgeConfig {
            collection: args.collection.clone(),
            ..KnowledgeConfig::default()
        };
        // Fails early if the embedder is not compiled in.
        let record = embedder::record(args.embedder_kind())?;
        if args.storage == StorageProfile::InMemory {
            let embedder = embedder::load(args).await?;
            if opts.receipt_key.is_none() {
                opts.receipt_key = match args.data_dir() {
                    Ok(dir) => Some(file_receipt_key(&dir.join("receipt.key"))?),
                    Err(e) => {
                        eprintln!("keep: warning: {e}; receipts use a key for this process only");
                        None
                    }
                };
            }
            let svc = Self::in_memory(opts);
            let knowledge = svc.knowledge;
            return Ok(Self {
                knowledge: knowledge.with_config(config).with_embedder(embedder),
                ..svc
            });
        }
        let url = args.postgres_url.as_deref().context(
            "KEEP_POSTGRES_URL (or --postgres-url) is required for this storage profile",
        )?;
        let with_ca;
        let url = match &args.postgres_ca {
            Some(ca) => {
                with_ca = with_root_cert(url, ca);
                with_ca.as_str()
            }
            None => url,
        };
        // Fail on a bad URL or TLS setting before touching the database.
        keep_postgres::connect::target(url).map_err(|e| anyhow!("KEEP_POSTGRES_URL: {e}"))?;
        let schema = args.postgres_schema.as_deref();
        let ledger = keep_ledger::PgMemoryLedger::connect_in_schema(url, schema)
            .await
            .context("connecting the Memory ledger")?;
        let journal = keep_ledger::PgErasureJournal::connect(url, schema, false)
            .await
            .context("connecting the Memory erasure journal")?;
        let memory = MemoryService::new(Arc::new(ledger), Arc::new(journal));
        let kurl = match schema {
            Some(s) => in_schema(url, s),
            None => url.to_owned(),
        };
        let knowledge = match args.storage {
            StorageProfile::PostgresVector => profile::postgres_vector(&kurl).await,
            StorageProfile::PostgresQdrant => {
                let Some(q) = args.qdrant_url.as_deref() else {
                    bail!("KEEP_QDRANT_URL (or --qdrant-url) is required for postgres-qdrant");
                };
                let ca =
                    match &args.qdrant_ca {
                        Some(path) => Some(std::fs::read(path).with_context(|| {
                            format!("reading KEEP_QDRANT_CA {}", path.display())
                        })?),
                        None => None,
                    };
                warn_plaintext_qdrant(q);
                profile::postgres_qdrant_with_ca(
                    &kurl,
                    q,
                    args.qdrant_api_key.clone(),
                    ca.as_deref(),
                )
                .await
            }
            StorageProfile::InMemory => unreachable!("handled above"),
        }
        .context("connecting Knowledge")?
        .with_config(config);
        let settings = Settings::connect(&kurl, schema).await?;
        check_embedder(&settings, knowledge.vectors(), &args.collection, &record).await?;
        if opts.receipt_key.is_none() {
            opts.receipt_key = Some(database_receipt_key(&settings).await?);
        }
        let embedder = embedder::load(args).await?;
        let knowledge = knowledge.with_embedder(embedder);
        Ok(Self::new(memory, knowledge, args.storage.as_str(), opts))
    }

    pub fn defaults(&self) -> &Scope {
        &self.defaults
    }

    pub fn info(&self) -> Value {
        json!({
            "name": "keep",
            "version": env!("CARGO_PKG_VERSION"),
            "storage": self.storage,
            "tokenizer": self.tokenizer.id(),
            "embeddings": self.knowledge.method(),
            "defaultScope": self.defaults,
            "model": self.model.as_ref().map(ModelSetup::info),
        })
    }

    /// Whether this process can serve requests: the ledger must answer.
    /// An unreachable vector store only degrades search to lexical
    /// matching, so it is reported but does not make the service unready.
    /// Never includes error text, which can carry connection details.
    pub async fn readiness(&self) -> (bool, Value) {
        let ledger = self.knowledge.probe_ledger().await.is_ok();
        let vectors = match self.knowledge.probe_vectors().await {
            None => "none",
            Some(Ok(())) => "ok",
            Some(Err(_)) => "degraded",
        };
        let mut info = self.info();
        info["status"] = json!(if ledger { "ready" } else { "unavailable" });
        info["checks"] = json!({
            "ledger": if ledger { "ok" } else { "unavailable" },
            "vectors": vectors,
        });
        (ledger, info)
    }

    /// Run due background jobs (retries, deferred vector deletions) of
    /// every scope. `keep serve` calls this on a timer.
    pub async fn run_due_jobs(&self) -> Result<keep_knowledge::engine::JobRun, ApiError> {
        Ok(self
            .knowledge
            .run_due_jobs(WORKER_CORPORA, WORKER_ROUNDS)
            .await?)
    }

    /// Run [`KeepService::run_due_jobs`] every `every` until the process
    /// exits. Failures are logged without detail and retried next tick.
    pub fn spawn_worker(
        self: &Arc<Self>,
        every: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let service = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let mut failing = false;
            loop {
                tick.tick().await;
                match service.run_due_jobs().await {
                    Ok(_) if failing => {
                        failing = false;
                        eprintln!("keep: background jobs running again");
                    }
                    Ok(_) => {}
                    Err(e) if !failing => {
                        failing = true;
                        eprintln!("keep: background jobs paused: {}", e.code.as_str());
                    }
                    Err(_) => {}
                }
            }
        })
    }

    /// Run one tool by name for the local caller (this process's MCP
    /// session).
    pub async fn call(&self, tool: &str, args: Value) -> Result<Value, ApiError> {
        self.call_as(tool, args, "local", None).await
    }

    /// Run one tool by name. Both transports go through here. `token`
    /// names the caller's credential and `session` its session, if known;
    /// they choose the evidence budget the call is charged to.
    pub async fn call_as(
        &self,
        tool: &str,
        args: Value,
        token: &str,
        session: Option<&str>,
    ) -> Result<Value, ApiError> {
        let started = Instant::now();
        let args = if args.is_null() { json!({}) } else { args };
        let caller = match self.budget.as_ref().map(|b| b.config().per) {
            Some(BudgetPer::Session) => format!("{token}/{}", session.unwrap_or("-")),
            _ => token.to_owned(),
        };
        let out = self.dispatch(tool, args, &caller).await;
        if self.log_calls {
            // Metadata only: never arguments or results.
            let outcome = match &out {
                Ok(_) => "ok",
                Err(e) => e.code.as_str(),
            };
            eprintln!(
                "keep: tool={tool} outcome={outcome} ms={}",
                started.elapsed().as_millis()
            );
        }
        out
    }

    async fn dispatch(&self, tool: &str, args: Value, caller: &str) -> Result<Value, ApiError> {
        match tool {
            "memory_context" => self.memory_context(parse(args)?).await,
            "memory_list" => self.memory_list(parse(args)?).await,
            "memory_read" => self.memory_read(parse(args)?).await,
            "memory_write" => self.memory_write(parse(args)?).await,
            "memory_history" => self.memory_history(parse(args)?).await,
            "memory_undo" => self.memory_undo(parse(args)?).await,
            "memory_update_metadata" => self.memory_update_metadata(parse(args)?).await,
            "memory_delete" => self.memory_delete(parse(args)?).await,
            "memory_restore" => self.memory_restore(parse(args)?).await,
            "memory_forget" => self.memory_forget(parse(args)?).await,
            "memory_export" => self.memory_export(parse(args)?).await,
            "knowledge_ingest" => self.knowledge_ingest(parse(args)?).await,
            "knowledge_search" => self.knowledge_search(parse(args)?, caller).await,
            "knowledge_fetch" => self.knowledge_fetch(parse(args)?, caller).await,
            "knowledge_index" => self.knowledge_index(parse(args)?).await,
            "knowledge_cite" => self.knowledge_cite(parse(args)?).await,
            "knowledge_traverse" => self.knowledge_traverse(parse(args)?).await,
            "knowledge_rebuild_projection" => self.knowledge_rebuild_projection(parse(args)?).await,
            "knowledge_status" => self.knowledge_status(parse(args)?).await,
            "knowledge_summarize" => self.knowledge_summarize(parse(args)?).await,
            "knowledge_forget_source" => self.knowledge_forget_source(parse(args)?).await,
            _ => Err(unknown_tool(tool)),
        }
    }

    // ------------------------------------------------------------ memory

    pub async fn memory_context(&self, req: MemoryContextReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let total = req.budget_tokens.unwrap_or(DEFAULT_CONTEXT_TOKENS);
        if total == 0 {
            return Err(ApiError::invalid("budget_tokens must be at least 1"));
        }
        if total > MAX_CONTEXT_TOKENS {
            return Err(budget_cap("budget_tokens", total, MAX_CONTEXT_TOKENS));
        }
        let budget = ContextBudget::for_total(total)?;
        let mut notes = Vec::new();
        let toc = if req.include_knowledge.unwrap_or(true) {
            match self.toc_for_slot(&scope, budget.knowledge).await {
                Ok(t) => t,
                Err(e) => {
                    notes.push(format!(
                        "knowledge contents unavailable: {}",
                        e.code.as_str()
                    ));
                    None
                }
            }
        } else {
            None
        };
        let has_toc = toc.is_some();
        let toc_fn = move |_: usize| toc.clone();
        let request = ContextRequest {
            budget,
            query: req.query,
            referenced_paths: req.referenced_paths,
            knowledge: has_toc.then_some(&toc_fn as &dyn KnowledgeToc),
            ..ContextRequest::default()
        };
        let personal = scope.project.is_some() && req.include_personal.unwrap_or(true);
        let mut block = if personal {
            let mine = self.tree(&Scope::new(scope.user.clone(), None)?).await?;
            self.memory
                .context_layers(
                    &[(&tree, "project"), (&mine, "personal")],
                    &request,
                    self.tokenizer.as_ref(),
                )
                .await?
        } else {
            self.memory
                .context(&tree, &request, self.tokenizer.as_ref())
                .await?
        };
        block.notes.extend(notes);
        let mut out = to_value(&block);
        out["scope"] = json!(scope.label());
        Ok(out)
    }

    pub async fn memory_list(&self, req: MemoryListReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let limit = req.limit.unwrap_or(MAX_LIST_PAGE);
        if limit == 0 || limit > MAX_LIST_PAGE {
            return Err(ApiError::invalid(format!(
                "limit must be between 1 and {MAX_LIST_PAGE}"
            )));
        }
        let query = ListQuery {
            category: req.category.map(|c| (!c.is_empty()).then_some(c)),
            after: req.after,
            limit,
            include_trashed: req.include_trashed,
        };
        let page = self.memory.list(&tree, &query).await?;
        let record = self.memory.tree(&tree).await?;
        let files: Vec<Value> = page
            .files
            .iter()
            .map(|f| {
                json!({
                    "memoryId": f.memory_id,
                    "path": f.meta.path,
                    "category": f.meta.category_id,
                    "displayName": f.meta.display_name,
                    "description": f.meta.description,
                    "revisionId": f.head_revision_id,
                    "version": f.version,
                    "sizeBytes": f.size_bytes,
                    "updatedAt": f.updated_at,
                    "trashed": f.deleted_at.is_some(),
                })
            })
            .collect();
        // A readable tree, grouped by category in display order.
        let mut categories = record.categories.clone();
        categories.sort_by_key(|c| c.sort_order);
        let mut tree_md = String::new();
        for c in &categories {
            let in_cat: Vec<_> = page
                .files
                .iter()
                .filter(|f| f.meta.category_id == c.id)
                .collect();
            if in_cat.is_empty() {
                continue;
            }
            tree_md.push_str(&format!("- {}\n", c.display_name));
            for f in in_cat {
                tree_md.push_str(&format!("  - {}", f.meta.path));
                if !f.meta.description.is_empty() {
                    tree_md.push_str(&format!(": {}", f.meta.description));
                }
                if f.deleted_at.is_some() {
                    tree_md.push_str(" (trashed)");
                }
                tree_md.push('\n');
            }
        }
        Ok(json!({
            "scope": scope.label(),
            "treeRevision": page.tree_revision,
            "total": page.total,
            "nextAfter": page.next_after,
            "tree": tree_md,
            "files": files,
        }))
    }

    pub async fn memory_read(&self, req: MemoryReadReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, false).await?;
        let view = self.memory.view(&tree, &id).await?;
        let head = view.head();
        let revision = match &req.revision_id {
            Some(r) => self.memory.view_revision(&tree, &id, r).await?,
            None => view.revision,
        };
        let mut out = json!({
            "scope": scope.label(),
            "memoryId": id,
            "path": revision.meta.path,
            "category": view.file.meta.category_id,
            "description": view.file.meta.description,
            "revisionId": revision.revision_id,
            "version": revision.version,
            "isHead": revision.revision_id == head.revision_id,
            "head": head_json(&head),
            "updatedAt": view.file.updated_at,
            "text": revision.text(),
        });
        if req.include_entries {
            out["entries"] = revision
                .entries
                .iter()
                .map(|e| json!({ "entryId": e.entry_id, "text": e.text }))
                .collect();
        }
        Ok(out)
    }

    pub async fn memory_write(&self, req: MemoryWriteReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let edit: Edit = match (req.text, req.edit) {
            (Some(text), None) => Edit::Write { text },
            (None, Some(e)) => e.into(),
            _ => return Err(ApiError::invalid("give exactly one of text or edit")),
        };
        let ctx = self.ctx(&scope, req.idempotency_key);
        let existing = match self.memory.view_by_path(&tree, &req.path).await {
            Ok(v) => Some(v),
            Err(MemoryError::NotFound { .. }) => None,
            Err(e) => return Err(e.into()),
        };
        let (out, created) = match (existing, req.expected_revision_id) {
            (None, Some(_)) => {
                return Err(ApiError::conflict(format!(
                    "there is no file at {} any more; it was moved, deleted or never existed",
                    req.path
                ))
                .with_reason("FILE_MISSING"));
            }
            (existing, None) => {
                let Edit::Write { text } = edit else {
                    return Err(match existing {
                        Some(v) => exists_conflict(&req.path, &v),
                        None => ApiError::not_found(format!("no file at {} to edit", req.path))
                            .with_reason("NOT_FOUND"),
                    });
                };
                let mut create = CreateFile::new(&req.path, text);
                if let Some(d) = req.description {
                    create = create.describe(d);
                }
                // With an idempotency key a retried create replays; without
                // one an existing path is a conflict.
                match self.memory.create(&ctx, &tree, &create).await {
                    Ok(out) => (out, true),
                    Err(MemoryError::PathExists(_)) => {
                        let v = self.memory.view_by_path(&tree, &req.path).await?;
                        return Err(exists_conflict(&req.path, &v));
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            (Some(view), Some(expected)) => {
                let head = self.expected_head(&tree, &view, &expected).await?;
                let id = &view.file.memory_id;
                let out = if req.merge {
                    self.memory
                        .edit_merging(&ctx, &tree, id, &head, &edit)
                        .await?
                } else {
                    self.memory.edit(&ctx, &tree, id, &head, &edit).await?
                };
                (out, false)
            }
        };
        let mut v = to_value(&out);
        v["scope"] = json!(scope.label());
        v["path"] = json!(req.path);
        v["created"] = json!(created && !out.replayed);
        v["revisionId"] = json!(out.head.revision_id);
        Ok(v)
    }

    pub async fn memory_history(&self, req: MemoryHistoryReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, true).await?;
        let revisions = self.memory.history(&tree, &id).await?;
        Ok(json!({ "scope": scope.label(), "memoryId": id, "revisions": revisions }))
    }

    pub async fn memory_undo(&self, req: MemoryUndoReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, false).await?;
        let view = self.memory.view(&tree, &id).await?;
        let head = self
            .expected_head(&tree, &view, &req.expected_revision_id)
            .await?;
        let ctx = self.ctx(&scope, None);
        let out = self
            .memory
            .undo(&ctx, &tree, &id, &head, req.target_revision_id.as_deref())
            .await?;
        Ok(write_json(&scope, &out))
    }

    pub async fn memory_update_metadata(
        &self,
        req: MemoryUpdateMetadataReq,
    ) -> Result<Value, ApiError> {
        if req.description.is_none() && req.display_name.is_none() {
            return Err(ApiError::invalid(
                "give a description or a display_name to change",
            ));
        }
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, false).await?;
        let view = self.memory.view(&tree, &id).await?;
        let head = self
            .expected_head(&tree, &view, &req.expected_revision_id)
            .await?;
        let ctx = self.ctx(&scope, None);
        let patch = MetadataPatch {
            description: req.description,
            display_name: req.display_name,
            entry_mode: None,
        };
        let out = self
            .memory
            .update_metadata(&ctx, &tree, &id, &head, &patch)
            .await?;
        let view = self.memory.view(&tree, &id).await?;
        let mut v = write_json(&scope, &out);
        v["path"] = json!(view.file.meta.path);
        v["description"] = json!(view.file.meta.description);
        v["displayName"] = json!(view.file.meta.display_name);
        Ok(v)
    }

    pub async fn memory_delete(&self, req: MemoryDeleteReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, false).await?;
        let view = self.memory.view(&tree, &id).await?;
        let head = self
            .expected_head(&tree, &view, &req.expected_revision_id)
            .await?;
        let ctx = self.ctx(&scope, None);
        let out = self.memory.delete(&ctx, &tree, &id, &head).await?;
        Ok(write_json(&scope, &out))
    }

    pub async fn memory_restore(&self, req: MemoryRestoreReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let id = self.resolve(&tree, &req, true).await?;
        let ctx = self.ctx(&scope, None);
        let out = self.memory.restore(&ctx, &tree, &id).await?;
        Ok(write_json(&scope, &out))
    }

    pub async fn memory_export(&self, req: MemoryExportReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let policy = match req.instruction_files.as_deref() {
            None | Some("refuse") => InstructionPolicy::Refuse,
            Some("skip") => InstructionPolicy::Skip,
            Some("allow_marked") => InstructionPolicy::AllowMarked,
            Some(other) => {
                return Err(ApiError::invalid(format!(
                    "instruction_files must be refuse, skip or allow_marked, not {other:?}"
                )));
            }
        };
        let tree = self.tree(&scope).await?;
        let bundle = export_bundle(&self.memory, &tree, policy).await?;
        Ok(json!({
            "scope": scope.label(),
            "files": bundle.files,
            "skippedInstructionFiles": bundle.report.skipped_instruction_files,
            "markedInstructionFiles": bundle.report.marked_instruction_files,
            "manifest": bundle.report.manifest,
        }))
    }

    pub async fn memory_forget(&self, req: MemoryForgetReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let tree = self.tree(&scope).await?;
        let names_file = req.path.is_some() || req.memory_id.is_some();
        let selector = match (req.text.clone(), req.entry_id.clone()) {
            (Some(_), Some(_)) => {
                return Err(ApiError::invalid("give text or entry_id, not both"));
            }
            (Some(text), None) => ForgetSelector::Text {
                memory_id: match names_file {
                    true => Some(self.resolve(&tree, &req, true).await?),
                    false => None,
                },
                text,
            },
            (None, Some(entry_id)) => ForgetSelector::Entry {
                memory_id: self.resolve(&tree, &req, true).await?,
                entry_id,
            },
            (None, None) => ForgetSelector::File {
                memory_id: self.resolve(&tree, &req, true).await?,
            },
        };
        let ctx = self.ctx(&scope, None);
        let receipt = self.memory.forget(&ctx, &tree, &selector).await?;
        Ok(json!({ "scope": scope.label(), "receipt": receipt }))
    }

    // ------------------------------------------------------------ knowledge

    pub async fn knowledge_ingest(&self, req: KnowledgeIngestReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let (default_kind, mut episodes) = match (req.text, req.episodes) {
            (Some(text), None) => {
                let ep = EpisodeInput {
                    id: "body".into(),
                    speaker: String::new(),
                    occurred_at: parse_time_opt("occurred_at", req.occurred_at.as_deref())?,
                    text,
                };
                (SourceKind::Document, vec![ep])
            }
            (None, Some(eps)) if !eps.is_empty() => {
                if req.occurred_at.is_some() {
                    return Err(ApiError::invalid(
                        "occurred_at applies to text; set it on each episode instead",
                    ));
                }
                let mut out = Vec::with_capacity(eps.len());
                for (i, e) in eps.into_iter().enumerate() {
                    out.push(EpisodeInput {
                        id: e.id.unwrap_or_else(|| format!("ep{:05}", i + 1)),
                        speaker: e.speaker.unwrap_or_else(|| "user".into()),
                        occurred_at: parse_time_opt("occurred_at", e.occurred_at.as_deref())?,
                        text: e.text,
                    });
                }
                (SourceKind::Conversation, out)
            }
            _ => {
                return Err(ApiError::invalid(
                    "give either text or a non-empty episodes list",
                ));
            }
        };
        if episodes.iter().all(|e| e.text.trim().is_empty()) {
            return Err(ApiError::invalid("nothing to ingest: the text is empty"));
        }
        let kind = match req.kind.as_deref() {
            None => default_kind,
            Some("document") => SourceKind::Document,
            Some("conversation") => SourceKind::Conversation,
            Some(other) => {
                return Err(ApiError::invalid(format!(
                    "kind must be document or conversation, not {other:?}"
                )));
            }
        };
        let source_id = match req.source_id {
            Some(id) if id.trim().is_empty() => {
                return Err(ApiError::invalid("source_id is empty"));
            }
            Some(id) => id,
            None => content_id(&episodes),
        };
        // Episode ids are unique across a corpus, so qualify them by source.
        for e in &mut episodes {
            e.id = format!("{source_id}#{}", e.id);
        }
        let corpus = scope.corpus();
        let partition = scope.partition();
        if let Some(existing) = self.knowledge.ledger().source(&corpus, &source_id).await?
            && existing.partition != partition
        {
            return Err(ApiError::conflict(format!(
                "source id {source_id:?} is already used in another scope; choose another id"
            ))
            .with_reason("SOURCE_IN_OTHER_SCOPE"));
        }
        let collection = req
            .collection
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_COLLECTION.into());
        let source = SourceInput {
            id: source_id.clone(),
            partition: partition.clone(),
            collection: collection.clone(),
            title: req.title.unwrap_or_else(|| source_id.clone()),
            kind,
            metadata: req.metadata.unwrap_or_default(),
        };
        let report = if req.replace {
            self.knowledge
                .ingest_revision(&corpus, source, episodes)
                .await?
        } else {
            self.knowledge.ingest(&corpus, source, episodes).await?
        };
        // The text is committed. Run this source's own jobs until the
        // deadline and leave the rest, and every other source's, to the
        // background worker. Enrichment that fails from here on leaves the
        // text captured; the response says so instead of reporting an error.
        let deadline = Instant::now() + self.ingest_job_budget;
        let jobs = if self.ingest_job_budget.is_zero() {
            Ok(Default::default())
        } else {
            self.knowledge
                .run_source_jobs(&corpus, &source_id, deadline)
                .await
        };
        let (jobs, jobs_error) = match jobs {
            Ok(run) => (Some(run), None),
            Err(e) => (None, Some(ApiError::from(e).code.as_str())),
        };
        let mut enrichment = match jobs_error {
            None => self.enrichment(&scope).await,
            Some(_) => json!({ "state": "unknown" }),
        };
        if let Some(code) = jobs_error {
            enrichment["error"] = json!(code);
        }
        Ok(json!({
            "scope": scope.label(),
            "sourceId": source_id,
            "partition": partition,
            "collection": collection,
            "captured": true,
            "replaced": report.replaced,
            "report": report,
            "jobs": jobs,
            "enrichment": enrichment,
        }))
    }

    pub async fn knowledge_status(&self, req: KnowledgeStatusReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let counts = self
            .knowledge
            .job_counts(&scope.corpus(), &scope.partition())
            .await?;
        Ok(json!({
            "scope": scope.label(),
            "enrichment": enrichment_of(&counts),
        }))
    }

    pub async fn knowledge_summarize(&self, req: KnowledgeSummarizeReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let access = read_access(&scope, req.source_ids, req.collections)?;
        let session = match (req.source_id, req.text, req.episodes) {
            (Some(id), None, None) => self.source_session(&scope, &access, &id).await?,
            (None, Some(text), None) => SessionInput {
                id: "text".into(),
                date: None,
                turns: text_turns("text", "", &text),
            },
            (None, None, Some(eps)) if !eps.is_empty() => {
                if eps.len() > MAX_SUMMARIZE_EPISODES {
                    return Err(ApiError::invalid(format!(
                        "episodes lists more than {MAX_SUMMARIZE_EPISODES} turns"
                    )));
                }
                let mut turns = Vec::with_capacity(eps.len());
                let mut date = None;
                for (i, e) in eps.into_iter().enumerate() {
                    let at = parse_time_opt("occurred_at", e.occurred_at.as_deref())?;
                    date = date.or(at);
                    turns.push(SessionTurn {
                        id: e.id.unwrap_or_else(|| format!("ep{:05}", i + 1)),
                        speaker: e.speaker.unwrap_or_else(|| "user".into()),
                        assistant: false,
                        text: e.text,
                    });
                }
                SessionInput {
                    id: "episodes".into(),
                    date,
                    turns,
                }
            }
            _ => {
                return Err(ApiError::invalid(
                    "give exactly one of source_id, text or a non-empty episodes list",
                ));
            }
        };
        let chars: usize = session.turns.iter().map(|t| t.text.chars().count()).sum();
        if chars > MAX_SUMMARIZE_CHARS {
            return Err(ApiError::new(
                ErrorCode::LimitExceeded,
                format!("{chars} characters to summarize; the limit is {MAX_SUMMARIZE_CHARS}"),
            )
            .with_reason("SUMMARIZE_TOO_LARGE"));
        }
        if session.turns.iter().all(|t| t.text.trim().is_empty()) {
            return Err(ApiError::invalid("nothing to summarize: the text is empty"));
        }
        let (Some(summarizer), Some(model)) = (&self.summarizer, &self.model) else {
            return Err(ApiError::new(
                ErrorCode::Unavailable,
                "no model is configured; set KEEP_MODEL_ENDPOINT and KEEP_MODEL_NAME",
            )
            .with_reason("MODEL_NOT_CONFIGURED"));
        };
        let summary = summarizer.summarize(&session).await.map_err(|e| match e {
            ProviderError::BudgetExhausted { .. } => {
                ApiError::new(ErrorCode::BudgetExceeded, format!("model budget: {e}"))
                    .with_reason("MODEL_BUDGET_EXHAUSTED")
            }
            e => ApiError::new(ErrorCode::Unavailable, format!("model call failed: {e}"))
                .with_reason("PROVIDER"),
        })?;
        Ok(json!({
            "scope": scope.label(),
            "summary": summary.text,
            "model": model.id,
            "usage": {
                "inputTokens": summary.usage.input_tokens,
                "outputTokens": summary.usage.output_tokens,
            },
        }))
    }

    /// A source's episodes as one session, or not found when `access` may
    /// not read it.
    async fn source_session(
        &self,
        scope: &Scope,
        access: &Access,
        id: &str,
    ) -> Result<SessionInput, ApiError> {
        let corpus = scope.corpus();
        let not_found = || ApiError::not_found(format!("source {id:?}"));
        let ledger = self.knowledge.ledger();
        match ledger.source(&corpus, id).await? {
            Some(s) if access.allows_source(&s) => {}
            _ => return Err(not_found()),
        }
        let episodes = ledger.source_episodes(&corpus, id).await?;
        if episodes.len() > MAX_SUMMARIZE_EPISODES * 10 {
            return Err(ApiError::new(
                ErrorCode::LimitExceeded,
                format!(
                    "source {id:?} has {} episodes; too many to summarize",
                    episodes.len()
                ),
            )
            .with_reason("SUMMARIZE_TOO_LARGE"));
        }
        let date = episodes.iter().find_map(|e| e.occurred_at);
        let turns = episodes
            .iter()
            .flat_map(|e| text_turns(&e.id, &e.speaker, &e.text))
            .collect();
        Ok(SessionInput {
            id: id.to_owned(),
            date,
            turns,
        })
    }

    /// The scope's enrichment state after a write, or `unknown` when it cannot
    /// be read. Finished jobs are not counted, so `jobs` has no `succeeded`;
    /// `knowledge_status` reports it.
    async fn enrichment(&self, scope: &Scope) -> Value {
        match self
            .knowledge
            .unfinished_job_counts(&scope.corpus(), &scope.partition())
            .await
        {
            Ok(counts) => {
                let mut value = enrichment_of(&counts);
                if let Some(jobs) = value["jobs"].as_object_mut() {
                    jobs.remove("succeeded");
                }
                value
            }
            Err(_) => json!({ "state": "unknown" }),
        }
    }

    pub async fn knowledge_search(
        &self,
        req: KnowledgeSearchReq,
        caller: &str,
    ) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        if req.query.trim().is_empty() {
            return Err(ApiError::invalid("query is empty"));
        }
        let request = SearchRequest {
            corpus: scope.corpus(),
            access: read_access(&scope, req.source_ids, req.collections)?,
            query: req.query,
            as_of: as_of(req.as_of.as_deref(), req.valid_at.as_deref())?,
            limit: req.limit.unwrap_or(10),
        };
        let (mut allowance, held) = self.allowance_for(caller, req.budget_tokens)?;
        let res = self.knowledge.search(&request, &mut allowance).await;
        self.settle(held, &allowance);
        let res = res?;
        let mut out = to_value(&res);
        out["scope"] = json!(scope.label());
        out["allowance"] = json!({ "limit": allowance.limit(), "used": allowance.used() });
        Ok(out)
    }

    pub async fn knowledge_fetch(
        &self,
        req: KnowledgeFetchReq,
        caller: &str,
    ) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let access = read_access(&scope, req.source_ids, req.collections)?;
        let as_of = as_of(req.as_of.as_deref(), req.valid_at.as_deref())?;
        let (mut allowance, held) = self.allowance_for(caller, req.budget_tokens)?;
        let res = self
            .knowledge
            .fetch(&scope.corpus(), &access, &req.id, as_of, &mut allowance)
            .await;
        self.settle(held, &allowance);
        let res = res?;
        let mut out = to_value(&res);
        out["scope"] = json!(scope.label());
        out["allowance"] = json!({ "limit": allowance.limit(), "used": allowance.used() });
        Ok(out)
    }

    pub async fn knowledge_index(&self, req: KnowledgeIndexReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        if req.level > 2 {
            return Err(ApiError::invalid("level must be 0, 1 or 2"));
        }
        let access = read_access(&scope, req.source_ids, req.collections)?;
        let request = IndexRequest {
            level: req.level,
            collection: req.collection,
            prefix: req.prefix,
            cursor: req.cursor,
            budget_tokens: req.budget_tokens,
        };
        let view = self
            .knowledge
            .index(&scope.corpus(), &access, &request)
            .await?;
        let mut out = to_value(&view);
        out["scope"] = json!(scope.label());
        Ok(out)
    }

    pub async fn knowledge_forget_source(
        &self,
        req: KnowledgeForgetSourceReq,
    ) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let corpus = scope.corpus();
        let not_found = || ApiError::not_found(format!("source {:?}", req.source_id));
        // Only the scope that wrote a source may erase it.
        match self
            .knowledge
            .ledger()
            .source(&corpus, &req.source_id)
            .await?
        {
            Some(s) if s.partition == scope.partition() => {}
            _ => return Err(not_found()),
        }
        let receipt = self
            .knowledge
            .forget_source(&corpus, &req.source_id)
            .await?
            .ok_or_else(not_found)?;
        Ok(json!({
            "scope": scope.label(),
            "sourceId": req.source_id,
            "receipt": receipt,
        }))
    }

    pub async fn knowledge_cite(&self, req: KnowledgeCiteReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let access = read_access(&scope, req.source_ids, req.collections)?;
        let citations = self
            .knowledge
            .cite(&scope.corpus(), &access, &req.id)
            .await?;
        Ok(json!({ "scope": scope.label(), "id": req.id, "citations": citations }))
    }

    pub async fn knowledge_traverse(&self, req: KnowledgeTraverseReq) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        if req.start.trim().is_empty() {
            return Err(ApiError::invalid("start is empty"));
        }
        let depth = req.depth.unwrap_or(1);
        if !(1..=2).contains(&depth) {
            return Err(ApiError::invalid("depth must be 1 or 2"));
        }
        let fanout = req.fanout.unwrap_or(10);
        if !(1..=25).contains(&fanout) {
            return Err(ApiError::invalid("fanout must be between 1 and 25"));
        }
        let as_of = as_of(req.as_of.as_deref(), req.valid_at.as_deref())?;
        let access = read_access(&scope, req.source_ids, req.collections)?;
        let n = self
            .knowledge
            .traverse(&scope.corpus(), &access, &req.start, depth, fanout, as_of)
            .await?;
        let mut out = to_value(&n);
        out["scope"] = json!(scope.label());
        Ok(out)
    }

    pub async fn knowledge_rebuild_projection(
        &self,
        req: KnowledgeRebuildProjectionReq,
    ) -> Result<Value, ApiError> {
        let scope = self.scope(req.scope.as_ref())?;
        let corpus = scope.corpus();
        let queued = self.knowledge.rebuild_projection(&corpus).await?;
        let jobs = self
            .knowledge
            .run_jobs(&corpus, rebuild_rounds(queued))
            .await?;
        Ok(json!({ "scope": scope.label(), "queued": queued, "jobs": jobs }))
    }

    // ------------------------------------------------------------ helpers

    /// The call's evidence allowance, cut to what is left of the caller's
    /// budget, with the reservation to settle afterwards.
    fn allowance_for(
        &self,
        caller: &str,
        tokens: Option<usize>,
    ) -> Result<(EvidenceAllowance, Option<Reservation>), ApiError> {
        let allowance = allowance(tokens)?;
        let Some(budget) = &self.budget else {
            return Ok((allowance, None));
        };
        let held = budget.reserve(caller, allowance.limit())?;
        Ok((EvidenceAllowance::new(held.tokens), Some(held)))
    }

    fn settle(&self, held: Option<Reservation>, allowance: &EvidenceAllowance) {
        if let (Some(budget), Some(held)) = (&self.budget, held) {
            budget.settle(held, allowance.used());
        }
    }

    fn scope(&self, arg: Option<&ScopeArg>) -> Result<Scope, ApiError> {
        Scope::resolve(arg, &self.defaults)
    }

    async fn tree(&self, scope: &Scope) -> Result<String, ApiError> {
        let id = scope.memory_tree();
        self.memory.ensure_tree(&id, &scope.memory_scope()).await?;
        Ok(id)
    }

    fn ctx(&self, scope: &Scope, key: Option<String>) -> OpCtx {
        let mut ctx = OpCtx::new(scope.user.clone());
        if let Some(a) = &self.agent {
            ctx = ctx.harness(a.clone());
        }
        if let Some(k) = key {
            ctx = ctx.with_key(k);
        }
        if self.profile_writes {
            ctx = ctx.pinning();
        }
        ctx
    }

    /// The file's id, by `memory_id` or by path. Trashed files are found by
    /// path only when `trashed` is set.
    async fn resolve(
        &self,
        tree: &str,
        req: &impl NamesFile,
        trashed: bool,
    ) -> Result<String, ApiError> {
        if let Some(id) = req.memory_id() {
            return Ok(id.to_owned());
        }
        let Some(path) = req.path() else {
            return Err(ApiError::invalid("give a path or a memory_id"));
        };
        match self.memory.view_by_path(tree, path).await {
            Ok(v) => return Ok(v.file.memory_id),
            Err(MemoryError::NotFound { .. }) if trashed => {}
            Err(e) => return Err(e.into()),
        }
        let mut after = None;
        loop {
            let page = self
                .memory
                .list(
                    tree,
                    &ListQuery {
                        after: after.clone(),
                        limit: MAX_LIST_PAGE,
                        include_trashed: true,
                        ..ListQuery::default()
                    },
                )
                .await?;
            if let Some(f) = page.files.iter().find(|f| f.meta.path == path) {
                return Ok(f.memory_id.clone());
            }
            match page.next_after {
                Some(a) => after = Some(a),
                None => return Err(ApiError::not_found(format!("file {path}"))),
            }
        }
    }

    /// The head the caller expects, from the revision id they last read.
    async fn expected_head(
        &self,
        tree: &str,
        view: &FileView,
        expected: &str,
    ) -> Result<Head, ApiError> {
        if expected == view.file.head_revision_id {
            return Ok(view.head());
        }
        let version = match self
            .memory
            .view_revision(tree, &view.file.memory_id, expected)
            .await
        {
            Ok(r) => r.version,
            // The engine then reports a conflict carrying the current head.
            Err(MemoryError::RevisionNotRetained(_)) => String::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Head {
            revision_id: expected.to_owned(),
            version,
        })
    }

    /// The largest level-0 Knowledge contents whose block fits `slot` tokens
    /// of the Memory context.
    async fn toc_for_slot(&self, scope: &Scope, slot: usize) -> Result<Option<String>, ApiError> {
        let tok = self.tokenizer.as_ref();
        let mut budget = slot.saturating_sub(toc_block_tokens("", tok));
        // JSON escaping can push the block over; shrink by the overrun.
        for _ in 0..4 {
            if budget == 0 {
                break;
            }
            let view = self
                .knowledge
                .index(
                    &scope.corpus(),
                    &scope.access(),
                    &IndexRequest {
                        level: 0,
                        budget_tokens: Some(budget),
                        ..IndexRequest::default()
                    },
                )
                .await?;
            if view.entries.is_empty() || view.text.trim().is_empty() {
                return Ok(None);
            }
            let cost = toc_block_tokens(&view.text, tok);
            if cost <= slot {
                return Ok(Some(view.text));
            }
            budget = budget.saturating_sub(cost - slot);
        }
        Ok(None)
    }
}

/// Most ids or collections one read may name.
const MAX_READ_FILTER: usize = 1000;

/// The scope's read access, narrowed to the requested sources and
/// collections.
fn read_access(
    scope: &Scope,
    source_ids: Option<Vec<String>>,
    collections: Option<Vec<String>>,
) -> Result<Access, ApiError> {
    let mut access = scope.access();
    for (field, list) in [("source_ids", &source_ids), ("collections", &collections)] {
        if list.as_ref().is_some_and(|l| l.len() > MAX_READ_FILTER) {
            return Err(ApiError::invalid(format!(
                "{field} lists more than {MAX_READ_FILTER} entries"
            )));
        }
    }
    if let Some(ids) = source_ids {
        access = access.only_sources(ids);
    }
    if let Some(names) = collections {
        access = access.only_collections(names);
    }
    Ok(access)
}

fn parse<T: DeserializeOwned>(args: Value) -> Result<T, ApiError> {
    serde_json::from_value(args)
        .map_err(|e| ApiError::invalid(format!("invalid arguments: {e}")).with_reason("ARGUMENTS"))
}

fn to_value<T: serde::Serialize>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

fn head_json(h: &Head) -> Value {
    json!({ "revisionId": h.revision_id, "version": h.version })
}

fn write_json(scope: &Scope, out: &keep_memory::service::WriteOutcome) -> Value {
    let mut v = to_value(out);
    v["scope"] = json!(scope.label());
    v["revisionId"] = json!(out.head.revision_id);
    v
}

fn exists_conflict(path: &str, v: &FileView) -> ApiError {
    ApiError::conflict(format!(
        "{path} already exists; read it and pass its revisionId as expected_revision_id"
    ))
    .with_reason("EXPECTED_REVISION_REQUIRED")
    .with_details(json!({ "memoryId": v.file.memory_id, "current": head_json(&v.head()) }))
}

fn budget_cap(what: &str, asked: usize, cap: usize) -> ApiError {
    ApiError::new(
        ErrorCode::BudgetExceeded,
        format!("{what} is {asked}; the most allowed is {cap}"),
    )
    .with_details(json!({ "requested": asked, "max": cap }))
}

/// Enough job rounds (64 jobs each) to drain a rebuild, plus slack.
fn rebuild_rounds(queued: usize) -> usize {
    queued.div_ceil(64) + REBUILD_EXTRA_ROUNDS
}

fn allowance(tokens: Option<usize>) -> Result<EvidenceAllowance, ApiError> {
    let tokens = tokens.unwrap_or(DEFAULT_EVIDENCE_TOKENS);
    if tokens == 0 {
        return Err(ApiError::invalid("budget_tokens must be at least 1"));
    }
    if tokens > MAX_EVIDENCE_TOKENS {
        return Err(budget_cap("budget_tokens", tokens, MAX_EVIDENCE_TOKENS));
    }
    Ok(EvidenceAllowance::new(tokens))
}

fn as_of(known: Option<&str>, valid: Option<&str>) -> Result<AsOf, ApiError> {
    Ok(AsOf {
        known_at: parse_time_opt("as_of", known)?,
        valid_at: parse_time_opt("valid_at", valid)?,
    })
}

/// RFC 3339, or a bare `YYYY-MM-DD` meaning midnight UTC.
pub fn parse_time(field: &str, s: &str) -> Result<DateTime<Utc>, ApiError> {
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.with_timezone(&Utc));
    }
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|t| t.and_utc())
        .ok_or_else(|| ApiError::invalid(format!("{field} must be RFC 3339 or YYYY-MM-DD")))
}

fn parse_time_opt(field: &str, s: Option<&str>) -> Result<Option<DateTime<Utc>>, ApiError> {
    s.map(|s| parse_time(field, s)).transpose()
}

/// A stable id for content ingested without one.
fn content_id(episodes: &[EpisodeInput]) -> String {
    let mut h = Sha256::new();
    for e in episodes {
        for part in [e.id.as_str(), e.speaker.as_str(), e.text.as_str()] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
    }
    format!("src_{}", &hex::encode(h.finalize())[..24])
}

/// Tokens the Knowledge block costs inside the Memory context.
fn toc_block_tokens(toc: &str, tokenizer: &dyn Tokenizer) -> usize {
    let body = json!({ "text": toc });
    tokenizer.count(&format!("\n{LABEL_KNOWLEDGE}\n{body}"))
}

/// `complete`, `pending` while jobs remain, or `failed` once some gave up.
/// Failed enrichment never loses the captured text.
fn enrichment_of(counts: &JobCounts) -> Value {
    let state = if counts.outstanding() > 0 {
        "pending"
    } else if counts.failed > 0 {
        "failed"
    } else {
        "complete"
    };
    json!({ "state": state, "jobs": counts })
}

/// Add `sslrootcert` to a Postgres URL (or key=value string).
fn with_root_cert(url: &str, ca: &std::path::Path) -> String {
    let path = ca.display().to_string();
    if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        let sep = if url.contains('?') { '&' } else { '?' };
        let mut enc = String::new();
        for b in path.bytes() {
            if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
                enc.push(b as char);
            } else {
                enc.push_str(&format!("%{b:02X}"));
            }
        }
        format!("{url}{sep}sslrootcert={enc}")
    } else {
        format!("{url} sslrootcert='{path}'")
    }
}

/// Qdrant over plain HTTP is fine on loopback or a private container
/// network; anywhere else its API key and vectors travel unencrypted.
fn warn_plaintext_qdrant(url: &str) {
    let Some(rest) = url.strip_prefix("http://") else {
        return;
    };
    let host = rest.split(['/', ':']).next().unwrap_or_default();
    let local = host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
        || !host.contains('.');
    if !local {
        eprintln!(
            "keep: warning: KEEP_QDRANT_URL uses plain HTTP to {host}; use https:// for a remote Qdrant"
        );
    }
}

/// A Postgres URL whose connections use `schema` first.
pub fn in_schema(url: &str, schema: &str) -> String {
    let sep = if url.contains('?') { '&' } else { '?' };
    format!("{url}{sep}options=-csearch_path%3D{schema},public")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_accept_rfc3339_and_dates() {
        assert_eq!(
            parse_time("t", "2025-03-04").unwrap().to_rfc3339(),
            "2025-03-04T00:00:00+00:00"
        );
        assert_eq!(
            parse_time("t", "2025-03-04T10:00:00+02:00")
                .unwrap()
                .to_rfc3339(),
            "2025-03-04T08:00:00+00:00"
        );
        assert_eq!(
            parse_time("t", "March").unwrap_err().code,
            ErrorCode::Invalid
        );
    }

    #[test]
    fn evidence_budgets_above_the_cap_are_refused_not_clamped() {
        let e = allowance(Some(MAX_EVIDENCE_TOKENS + 1)).unwrap_err();
        assert_eq!(e.code, ErrorCode::BudgetExceeded);
        assert_eq!(allowance(None).unwrap().limit(), DEFAULT_EVIDENCE_TOKENS);
    }
}
