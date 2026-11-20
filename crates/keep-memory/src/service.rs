//! The MEMORY service: every operation, with compare-and-swap, limits,
//! retention, idempotency and forget.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Duration, Utc};
use keep_core::memory::{BehaviorHint, Category, MemoryLimits};
use keep_core::{canonical_json, sha256_hex};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::align::{AlignCtx, align};
use crate::codec::{self, EntryMode};
use crate::digest::hmac_sha256_hex;
use crate::error::{LimitKind, MemoryError, Result};
use crate::ledger::{ErasureJournal, LedgerTx, MemoryLedger};
use crate::model::{
    Annotation, ChangeKind, Entry, ErasureRecord, FileMeta, FileRecord, Head, Receipt,
    ReceiptOutcome, Revision, TreeRecord,
};
use crate::path;

// ---------------------------------------------------------------- plumbing

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A clock tests and benchmarks move by hand.
#[derive(Debug)]
pub struct ManualClock(std::sync::Mutex<DateTime<Utc>>);

impl ManualClock {
    pub fn new(start: DateTime<Utc>) -> Self {
        Self(std::sync::Mutex::new(start))
    }

    pub fn advance(&self, by: Duration) {
        let mut t = self.0.lock().unwrap_or_else(|e| e.into_inner());
        *t += by;
    }

    pub fn set(&self, to: DateTime<Utc>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = to;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> DateTime<Utc> {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub trait IdSource: Send + Sync {
    /// A fresh identifier such as `mem_…`, `rev_…` or `ent_…`.
    fn next(&self, prefix: &str) -> String;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RandomIds;

impl IdSource for RandomIds {
    fn next(&self, prefix: &str) -> String {
        format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
    }
}

/// Deterministic identifiers for reproducible benchmarks and tests.
#[derive(Debug, Default)]
pub struct SequentialIds(AtomicU64);

impl IdSource for SequentialIds {
    fn next(&self, prefix: &str) -> String {
        let n = self.0.fetch_add(1, Ordering::Relaxed) + 1;
        format!("{prefix}_{n:012}")
    }
}

/// Stored times keep microsecond precision so every backend agrees.
pub fn micros(t: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(t.timestamp_micros()).unwrap_or(t)
}

/// How long history and trash are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retention {
    /// Prior revisions kept per file; the current head is not counted.
    pub max_revisions: usize,
    pub max_revision_age: Duration,
    pub trash: Duration,
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            max_revisions: 100,
            max_revision_age: Duration::days(30),
            trash: Duration::days(7),
        }
    }
}

/// Who is acting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actor {
    pub principal: String,
    pub harness: Option<String>,
}

/// Per-operation context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpCtx {
    pub actor: Actor,
    pub idempotency_key: Option<String>,
    /// Whether this caller may place files in a profile category, which is
    /// considered for every context bundle.
    pub can_pin_profile: bool,
    /// Sources the written text derives from. They become protected lineage
    /// on the entries this operation writes.
    pub dependencies: BTreeSet<String>,
}

impl OpCtx {
    pub fn new(principal: impl Into<String>) -> Self {
        Self {
            actor: Actor {
                principal: principal.into(),
                harness: None,
            },
            idempotency_key: None,
            can_pin_profile: false,
            dependencies: BTreeSet::new(),
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    pub fn pinning(mut self) -> Self {
        self.can_pin_profile = true;
        self
    }

    pub fn harness(mut self, harness: impl Into<String>) -> Self {
        self.actor.harness = Some(harness.into());
        self
    }

    pub fn depends_on(mut self, source: impl Into<String>) -> Self {
        self.dependencies.insert(source.into());
        self
    }
}

// ---------------------------------------------------------------- requests

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateFile {
    pub path: String,
    pub text: String,
    pub display_name: Option<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub entry_mode: EntryMode,
    /// Fail unless the tree is still at this revision.
    pub expected_tree_revision: Option<u64>,
}

impl CreateFile {
    pub fn new(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            text: text.into(),
            ..Self::default()
        }
    }

    pub fn mode(mut self, mode: EntryMode) -> Self {
        self.entry_mode = mode;
        self
    }

    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Hunk {
    pub old: String,
    pub new: String,
}

/// A text edit. All of them operate on the visible Markdown.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Edit {
    /// Replace `old` with `new`; `old` must match once unless `replace_all`.
    StrReplace {
        old: String,
        new: String,
        #[serde(default)]
        replace_all: bool,
    },
    /// Insert `text` before 0-based `line`; the line count appends.
    Insert { line: usize, text: String },
    /// Apply hunks in order; each `old` must match exactly once at its turn.
    Patch { hunks: Vec<Hunk> },
    /// Replace the whole file.
    Write { text: String },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataPatch {
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub entry_mode: Option<EntryMode>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ListQuery {
    /// `None` lists all; `Some(None)` lists uncategorised files.
    pub category: Option<Option<String>>,
    /// Return files with paths after this one.
    pub after: Option<String>,
    /// Page size; 0 means 100.
    pub limit: usize,
    pub include_trashed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ForgetSelector {
    /// The whole file, all its history and its trash copy.
    File { memory_id: String },
    /// One entry across every revision of its file.
    Entry { memory_id: String, entry_id: String },
    /// Every occurrence of exact text, in one file or the whole tree.
    Text {
        memory_id: Option<String>,
        text: String,
    },
}

impl ForgetSelector {
    fn kind(&self) -> &'static str {
        match self {
            ForgetSelector::File { .. } => "file",
            ForgetSelector::Entry { .. } => "entry",
            ForgetSelector::Text { .. } => "text",
        }
    }
}

// ---------------------------------------------------------------- results

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteOutcome {
    pub memory_id: String,
    pub head: Head,
    pub seq: u64,
    pub tree_revision: u64,
    /// False when the request changed nothing.
    pub changed: bool,
    /// True when this is a replayed idempotent request.
    pub replayed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileView {
    pub file: FileRecord,
    pub revision: Revision,
}

impl FileView {
    pub fn text(&self) -> String {
        self.revision.text()
    }

    pub fn head(&self) -> Head {
        self.file.head()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListPage {
    pub files: Vec<FileRecord>,
    pub next_after: Option<String>,
    /// Files matching the filter across all pages.
    pub total: usize,
    pub tree_revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevisionSummary {
    pub revision_id: String,
    pub seq: u64,
    pub change: ChangeKind,
    pub actor: String,
    pub harness: Option<String>,
    pub created_at: DateTime<Utc>,
    pub version: String,
    pub path: String,
    pub size_bytes: usize,
    pub is_head: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForgetOutcome {
    pub erasure_seq: u64,
    pub generation: u64,
    pub files_affected: Vec<String>,
    pub files_purged: Vec<String>,
    pub revisions_purged: usize,
    pub receipts_retired: usize,
    /// `journaled` when the erasure journal survives a ledger restore,
    /// otherwise `single_store`.
    pub recovery: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaintenanceOutcome {
    pub files_purged: usize,
    pub revisions_purged: usize,
}

// ---------------------------------------------------------------- service

pub struct MemoryService {
    ledger: Arc<dyn MemoryLedger>,
    journal: Arc<dyn ErasureJournal>,
    clock: Arc<dyn Clock>,
    ids: Arc<dyn IdSource>,
    limits: MemoryLimits,
    retention: Retention,
    receipt_key: Vec<u8>,
}

/// What an operation produces before the common commit path.
enum Content {
    Same,
    Text(String),
    Entries(Vec<Entry>),
    /// New text aligned against these entries instead of the head's.
    Realign(Vec<Entry>, String),
}

enum ReceiptCheck {
    Replay(WriteOutcome),
    Fresh(Option<(String, String)>), // (key, digest)
}

impl MemoryService {
    pub fn new(ledger: Arc<dyn MemoryLedger>, journal: Arc<dyn ErasureJournal>) -> Self {
        let key = format!("{}{}", uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        Self {
            ledger,
            journal,
            clock: Arc::new(SystemClock),
            ids: Arc::new(RandomIds),
            limits: MemoryLimits::default(),
            retention: Retention::default(),
            receipt_key: key.into_bytes(),
        }
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn with_ids(mut self, ids: Arc<dyn IdSource>) -> Self {
        self.ids = ids;
        self
    }

    pub fn with_limits(mut self, limits: MemoryLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_retention(mut self, retention: Retention) -> Self {
        self.retention = retention;
        self
    }

    /// The server secret used for receipt digests. Keep it stable across
    /// restarts so idempotent retries keep working.
    pub fn with_receipt_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.receipt_key = key.into();
        self
    }

    pub fn limits(&self) -> &MemoryLimits {
        &self.limits
    }

    fn now(&self) -> DateTime<Utc> {
        micros(self.clock.now())
    }

    // ------------------------------------------------------------ trees

    /// Create the tree if it does not exist and return it.
    pub async fn ensure_tree(&self, tree_id: &str, scope: &str) -> Result<TreeRecord> {
        let mut tx = self.ledger.begin().await?;
        if let Some(t) = tx.tree(tree_id).await? {
            return Ok(t);
        }
        let tree = TreeRecord {
            tree_id: tree_id.to_owned(),
            scope: scope.to_owned(),
            tree_revision: 0,
            erasure_generation: 0,
            categories: Category::defaults(),
            created_at: self.now(),
        };
        tx.put_tree(&tree).await?;
        tx.commit().await?;
        Ok(tree)
    }

    pub async fn tree(&self, tree_id: &str) -> Result<TreeRecord> {
        let mut tx = self.ledger.begin().await?;
        load_tree(tx.as_mut(), tree_id).await
    }

    // ------------------------------------------------------------ reads

    pub async fn view(&self, tree_id: &str, memory_id: &str) -> Result<FileView> {
        let mut tx = self.ledger.begin().await?;
        let file = self.live_file(tx.as_mut(), tree_id, memory_id).await?;
        let revision = load_revision(tx.as_mut(), &file.head_revision_id).await?;
        Ok(FileView { file, revision })
    }

    pub async fn view_by_path(&self, tree_id: &str, path: &str) -> Result<FileView> {
        let mut tx = self.ledger.begin().await?;
        let file = tx
            .files(tree_id, false)
            .await?
            .into_iter()
            .find(|f| f.meta.path == path)
            .ok_or_else(|| MemoryError::not_found(format!("file {path}")))?;
        let revision = load_revision(tx.as_mut(), &file.head_revision_id).await?;
        Ok(FileView { file, revision })
    }

    pub async fn list(&self, tree_id: &str, query: &ListQuery) -> Result<ListPage> {
        let mut tx = self.ledger.begin().await?;
        let tree = load_tree(tx.as_mut(), tree_id).await?;
        let now = self.now();
        let matching: Vec<FileRecord> = tx
            .files(tree_id, query.include_trashed)
            .await?
            .into_iter()
            .filter(|f| !self.trash_expired(f, now))
            .filter(|f| {
                query
                    .category
                    .as_ref()
                    .is_none_or(|c| *c == f.meta.category_id)
            })
            .collect();
        let total = matching.len();
        let limit = if query.limit == 0 { 100 } else { query.limit };
        let files: Vec<FileRecord> = matching
            .into_iter()
            .filter(|f| query.after.as_ref().is_none_or(|a| f.meta.path > *a))
            .take(limit + 1)
            .collect();
        let mut files = files;
        let next_after = if files.len() > limit {
            files.truncate(limit);
            files.last().map(|f| f.meta.path.clone())
        } else {
            None
        };
        Ok(ListPage {
            files,
            next_after,
            total,
            tree_revision: tree.tree_revision,
        })
    }

    /// Every live file with its head revision, ordered by path.
    pub async fn snapshot(&self, tree_id: &str) -> Result<(TreeRecord, Vec<FileView>)> {
        let mut tx = self.ledger.begin().await?;
        let tree = load_tree(tx.as_mut(), tree_id).await?;
        let mut out = Vec::new();
        for file in tx.files(tree_id, false).await? {
            let revision = load_revision(tx.as_mut(), &file.head_revision_id).await?;
            out.push(FileView { file, revision });
        }
        Ok((tree, out))
    }

    /// Revision summaries, newest first, including the head.
    pub async fn history(&self, tree_id: &str, memory_id: &str) -> Result<Vec<RevisionSummary>> {
        let mut tx = self.ledger.begin().await?;
        let file = self.any_file(tx.as_mut(), tree_id, memory_id).await?;
        let mut out: Vec<RevisionSummary> = tx
            .revisions(memory_id)
            .await?
            .into_iter()
            .map(|r| RevisionSummary {
                is_head: r.revision_id == file.head_revision_id,
                size_bytes: r.rendered_len(),
                path: r.meta.path.clone(),
                revision_id: r.revision_id,
                seq: r.seq,
                change: r.change,
                actor: r.actor,
                harness: r.harness,
                created_at: r.created_at,
                version: r.version,
            })
            .collect();
        out.reverse();
        Ok(out)
    }

    pub async fn view_revision(
        &self,
        tree_id: &str,
        memory_id: &str,
        revision_id: &str,
    ) -> Result<Revision> {
        let mut tx = self.ledger.begin().await?;
        self.any_file(tx.as_mut(), tree_id, memory_id).await?;
        match tx.revision(revision_id).await? {
            Some(r) if r.memory_id == memory_id => Ok(r),
            _ => Err(MemoryError::RevisionNotRetained(revision_id.to_owned())),
        }
    }

    // ------------------------------------------------------------ writes

    pub async fn create(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        req: &CreateFile,
    ) -> Result<WriteOutcome> {
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let request = json!({"op": "create", "req": req});
        let digest = match self
            .check_receipt(tx.as_mut(), ctx, tree_id, &request)
            .await?
        {
            ReceiptCheck::Replay(o) => return Ok(o),
            ReceiptCheck::Fresh(d) => d,
        };
        if let Some(expected) = req.expected_tree_revision
            && expected != tree.tree_revision
        {
            return Err(MemoryError::TreeConflict {
                current: tree.tree_revision,
            });
        }
        let parsed = path::validate_path(&req.path)?;
        path::validate_description(&req.description)?;
        let (category_id, project_id) = path::infer(&parsed)?;
        let now = self.now();
        let meta = FileMeta {
            path: req.path.clone(),
            category_id,
            display_name: req
                .display_name
                .clone()
                .unwrap_or_else(|| path::default_display_name(&parsed)),
            description: req.description.clone(),
            project_id,
            entry_mode: req.entry_mode,
        };
        let memory_id = self.ids.next("mem");
        let (file, _) = self
            .commit_change(
                tx.as_mut(),
                &mut tree,
                ctx,
                now,
                Draft {
                    memory_id: memory_id.clone(),
                    current: None,
                    head: None,
                    meta,
                    content: Content::Text(req.text.clone()),
                    change: ChangeKind::Create,
                    parents: Vec::new(),
                },
            )
            .await?;
        let outcome = outcome(&file, &tree, true);
        self.store_receipt(tx.as_mut(), ctx, tree_id, "create", digest, &outcome, now)
            .await?;
        tx.commit().await?;
        Ok(outcome)
    }

    pub async fn edit(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        edit: &Edit,
    ) -> Result<WriteOutcome> {
        let request =
            json!({"op": "edit", "memory": memory_id, "expected": expected, "edit": edit});
        self.edit_with_request(ctx, tree_id, memory_id, expected, edit, request)
            .await
    }

    async fn edit_with_request(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        edit: &Edit,
        request: Value,
    ) -> Result<WriteOutcome> {
        let edit = edit.clone();
        self.mutate(ctx, tree_id, memory_id, expected, request, move |head| {
            let text = apply_edit(&head.text(), &edit)?;
            Ok((head.meta.clone(), Content::Text(text), ChangeKind::Edit))
        })
        .await
    }

    /// Like [`Self::edit`], but when `expected` is stale and the edit touches
    /// only entries the newer saves left alone, replay it onto the current
    /// head as a merge instead of failing. Overlapping changes return
    /// [`MemoryError::MergeConflict`]; an unretained base returns the
    /// original version conflict.
    pub async fn edit_merging(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        edit: &Edit,
    ) -> Result<WriteOutcome> {
        // Both attempts share one request, so a retried key replays either.
        let request = json!({"op": "edit", "memory": memory_id, "expected": expected,
            "edit": edit, "merge": true});
        let current = match self
            .edit_with_request(ctx, tree_id, memory_id, expected, edit, request.clone())
            .await
        {
            Err(MemoryError::VersionConflict { current }) => current,
            other => return other,
        };
        let conflict = || MemoryError::VersionConflict {
            current: current.clone(),
        };
        let (base, theirs) = {
            let mut tx = self.ledger.begin().await?;
            let base = tx.revision(&expected.revision_id).await?;
            let theirs = tx.revision(&current.revision_id).await?;
            drop(tx);
            match (base, theirs) {
                (Some(b), Some(t)) if b.memory_id == memory_id && t.memory_id == memory_id => {
                    (b, t)
                }
                _ => return Err(conflict()),
            }
        };
        if base.version != expected.version {
            return Err(conflict());
        }
        let ours = apply_edit(&base.text(), edit)?;
        let merged = crate::merge::merge_text(
            &base.entries,
            &ours,
            &theirs.entries,
            theirs.meta.entry_mode,
            self.now(),
        )?;
        self.mutate(ctx, tree_id, memory_id, &current, request, move |head| {
            Ok((head.meta.clone(), Content::Text(merged), ChangeKind::Merge))
        })
        .await
    }

    /// Replace a file's text with imported text, recorded as an import.
    pub async fn import_write(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        text: &str,
    ) -> Result<WriteOutcome> {
        let request =
            json!({"op": "import", "memory": memory_id, "expected": expected, "text": text});
        let text = text.to_owned();
        self.mutate(ctx, tree_id, memory_id, expected, request, move |head| {
            Ok((head.meta.clone(), Content::Text(text), ChangeKind::Import))
        })
        .await
    }

    /// Replace the descriptive annotation of one entry.
    pub async fn annotate(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        entry_id: &str,
        annotation: &Annotation,
    ) -> Result<WriteOutcome> {
        validate_annotation(annotation)?;
        let request = json!({"op": "annotate", "memory": memory_id, "expected": expected,
            "entry": entry_id, "annotation": annotation});
        let entry_id = entry_id.to_owned();
        let annotation = annotation.clone();
        let actor = ctx.actor.principal.clone();
        let now = self.now();
        self.mutate(ctx, tree_id, memory_id, expected, request, move |head| {
            let mut entries = head.entries.clone();
            let e = entries
                .iter_mut()
                .find(|e| e.entry_id == entry_id)
                .ok_or_else(|| MemoryError::not_found(format!("entry {entry_id}")))?;
            e.annotation = annotation;
            e.updated_at = now;
            e.lineage.updated_by = actor;
            Ok((
                head.meta.clone(),
                Content::Entries(entries),
                ChangeKind::Annotate,
            ))
        })
        .await
    }

    pub async fn update_metadata(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        patch: &MetadataPatch,
    ) -> Result<WriteOutcome> {
        if let Some(d) = &patch.description {
            path::validate_description(d)?;
        }
        if let Some(n) = &patch.display_name {
            validate_display_name(n)?;
        }
        let request =
            json!({"op": "metadata", "memory": memory_id, "expected": expected, "patch": patch});
        let patch = patch.clone();
        self.mutate(ctx, tree_id, memory_id, expected, request, move |head| {
            let mut meta = head.meta.clone();
            if let Some(n) = patch.display_name {
                meta.display_name = n;
            }
            if let Some(d) = patch.description {
                meta.description = d;
            }
            let content = match patch.entry_mode {
                Some(m) if m != meta.entry_mode => {
                    meta.entry_mode = m;
                    Content::Text(head.text())
                }
                _ => Content::Same,
            };
            Ok((meta, content, ChangeKind::Metadata))
        })
        .await
    }

    pub async fn rename(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        new_path: &str,
    ) -> Result<WriteOutcome> {
        let parsed = path::validate_path(new_path)?;
        let (category_id, project_id) = path::infer(&parsed)?;
        let request =
            json!({"op": "rename", "memory": memory_id, "expected": expected, "path": new_path});
        let new_path = new_path.to_owned();
        self.mutate(ctx, tree_id, memory_id, expected, request, move |head| {
            let mut meta = head.meta.clone();
            if meta.display_name == path::default_display_name(&path::validate_path(&meta.path)?) {
                meta.display_name = path::default_display_name(&parsed);
            }
            meta.path = new_path;
            meta.category_id = category_id;
            meta.project_id = project_id;
            Ok((meta, Content::Same, ChangeKind::Rename))
        })
        .await
    }

    /// Restore a retained revision as a new revision. Defaults to the head's
    /// parent. History is never rewritten.
    pub async fn undo(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        target: Option<&str>,
    ) -> Result<WriteOutcome> {
        let target_id = match target {
            Some(t) => t.to_owned(),
            None => {
                let mut tx = self.ledger.begin().await?;
                let file = self.live_file(tx.as_mut(), tree_id, memory_id).await?;
                let head = load_revision(tx.as_mut(), &file.head_revision_id).await?;
                let parent = head.parents.first().ok_or(MemoryError::NothingToUndo)?;
                match tx.revision(parent).await? {
                    Some(r) => r.revision_id,
                    None => return Err(MemoryError::NothingToUndo),
                }
            }
        };
        let target = {
            let mut tx = self.ledger.begin().await?;
            match tx.revision(&target_id).await? {
                Some(r) if r.memory_id == memory_id => r,
                _ => return Err(MemoryError::RevisionNotRetained(target_id)),
            }
        };
        let request = json!({"op": "undo", "memory": memory_id, "expected": expected, "target": target.revision_id});
        self.mutate(ctx, tree_id, memory_id, expected, request, move |_head| {
            Ok((
                target.meta.clone(),
                Content::Entries(target.entries.clone()),
                ChangeKind::Undo,
            ))
        })
        .await
    }

    /// Move a file to the trash. Its path is freed immediately; it can be
    /// restored for the trash retention period.
    pub async fn delete(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
    ) -> Result<WriteOutcome> {
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let request = json!({"op": "delete", "memory": memory_id, "expected": expected});
        let digest = match self
            .check_receipt(tx.as_mut(), ctx, tree_id, &request)
            .await?
        {
            ReceiptCheck::Replay(o) => return Ok(o),
            ReceiptCheck::Fresh(d) => d,
        };
        let mut file = self.live_file(tx.as_mut(), tree_id, memory_id).await?;
        check_head(&file, expected)?;
        let now = self.now();
        file.deleted_at = Some(now);
        file.updated_at = now;
        tx.put_file(&file).await?;
        tree.tree_revision += 1;
        tx.put_tree(&tree).await?;
        let outcome = outcome(&file, &tree, true);
        self.store_receipt(tx.as_mut(), ctx, tree_id, "delete", digest, &outcome, now)
            .await?;
        tx.commit().await?;
        Ok(outcome)
    }

    /// Bring a trashed file back. Its old path must be free.
    pub async fn restore(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
    ) -> Result<WriteOutcome> {
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let request = json!({"op": "restore", "memory": memory_id});
        let digest = match self
            .check_receipt(tx.as_mut(), ctx, tree_id, &request)
            .await?
        {
            ReceiptCheck::Replay(o) => return Ok(o),
            ReceiptCheck::Fresh(d) => d,
        };
        let mut file = self.any_file(tx.as_mut(), tree_id, memory_id).await?;
        let now = self.now();
        if file.is_live() {
            return Err(MemoryError::NotInTrash);
        }
        if self.trash_expired(&file, now) {
            return Err(MemoryError::TrashExpired);
        }
        let live = tx.files(tree_id, false).await?;
        if live.len() >= self.limits.max_files_per_tree {
            return Err(MemoryError::Limit {
                kind: LimitKind::FilesPerTree,
                limit: self.limits.max_files_per_tree,
                actual: live.len() + 1,
            });
        }
        let total: usize = live.iter().map(|f| f.size_bytes).sum::<usize>() + file.size_bytes;
        if total > self.limits.max_rendered_bytes_per_tree {
            return Err(MemoryError::Limit {
                kind: LimitKind::RenderedBytesPerTree,
                limit: self.limits.max_rendered_bytes_per_tree,
                actual: total,
            });
        }
        let key = path::collision_key(&file.meta.path);
        if live
            .iter()
            .any(|f| path::collision_key(&f.meta.path) == key)
        {
            return Err(MemoryError::PathExists(file.meta.path.clone()));
        }
        self.check_profile_permission(ctx, &tree, None, &file.meta)?;
        file.deleted_at = None;
        file.updated_at = now;
        tx.put_file(&file).await?;
        tree.tree_revision += 1;
        tx.put_tree(&tree).await?;
        let outcome = outcome(&file, &tree, true);
        self.store_receipt(tx.as_mut(), ctx, tree_id, "restore", digest, &outcome, now)
            .await?;
        tx.commit().await?;
        Ok(outcome)
    }

    // ------------------------------------------------------------ forget

    /// Permanently erase content from every revision, the trash, annotations,
    /// descriptions and idempotency receipts. There is no undo. A
    /// content-free record is written to the erasure journal first.
    pub async fn forget(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        selector: &ForgetSelector,
    ) -> Result<ForgetOutcome> {
        if let ForgetSelector::Text { text, .. } = selector
            && text.is_empty()
        {
            return Err(MemoryError::InvalidRequest("forget text is empty".into()));
        }
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let now = self.now();
        let targets: Vec<FileRecord> = match selector {
            ForgetSelector::File { memory_id } | ForgetSelector::Entry { memory_id, .. } => {
                vec![self.any_file(tx.as_mut(), tree_id, memory_id).await?]
            }
            ForgetSelector::Text {
                memory_id: Some(m), ..
            } => vec![self.any_file(tx.as_mut(), tree_id, m).await?],
            ForgetSelector::Text {
                memory_id: None, ..
            } => tx.files(tree_id, true).await?,
        };

        let mut purged_revisions: Vec<String> = Vec::new();
        let mut files_affected = Vec::new();
        let mut files_purged = Vec::new();
        let mut new_heads: Vec<(FileRecord, Revision, Draft)> = Vec::new();
        for file in targets {
            let revisions = tx.revisions(&file.memory_id).await?;
            let head = revisions
                .iter()
                .find(|r| r.revision_id == file.head_revision_id)
                .cloned()
                .ok_or_else(|| MemoryError::Storage("file head is missing".into()))?;
            let (hit, purge_whole): (Vec<&Revision>, bool) = match selector {
                ForgetSelector::File { .. } => (revisions.iter().collect(), true),
                ForgetSelector::Entry { entry_id, .. } => {
                    let hit: Vec<&Revision> = revisions
                        .iter()
                        .filter(|r| r.entries.iter().any(|e| &e.entry_id == entry_id))
                        .collect();
                    if hit.is_empty() {
                        return Err(MemoryError::not_found(format!("entry {entry_id}")));
                    }
                    (hit, false)
                }
                ForgetSelector::Text { text, .. } => {
                    let hit: Vec<&Revision> =
                        revisions.iter().filter(|r| r.mentions(text)).collect();
                    // Text inside the path cannot be removed without renaming
                    // the file, so the whole file goes.
                    let whole = revisions
                        .iter()
                        .any(|r| r.meta.path.contains(text.as_str()));
                    (
                        if whole {
                            revisions.iter().collect()
                        } else {
                            hit
                        },
                        whole,
                    )
                }
            };
            if hit.is_empty() {
                continue;
            }
            files_affected.push(file.memory_id.clone());
            purged_revisions.extend(hit.iter().map(|r| r.revision_id.clone()));
            if purge_whole {
                files_purged.push(file.memory_id.clone());
                continue;
            }
            let head_hit = hit.iter().any(|r| r.revision_id == head.revision_id);
            if !head_hit {
                continue;
            }
            let (meta, content) = scrub(&head, selector);
            let draft = Draft {
                memory_id: file.memory_id.clone(),
                current: Some(file.clone()),
                head: Some(head.clone()),
                meta,
                content,
                change: ChangeKind::Forget,
                parents: Vec::new(),
            };
            new_heads.push((file, head, draft));
        }
        if files_affected.is_empty() {
            return Err(MemoryError::not_found("matching content"));
        }

        tree.erasure_generation += 1;
        tree.tree_revision += 1;
        let erasure_seq = self
            .journal
            .append(ErasureRecord {
                seq: 0,
                tree_id: tree_id.to_owned(),
                memory_id: match selector {
                    ForgetSelector::File { memory_id }
                    | ForgetSelector::Entry { memory_id, .. } => Some(memory_id.clone()),
                    ForgetSelector::Text { memory_id, .. } => memory_id.clone(),
                },
                selector: selector.kind().to_owned(),
                generation: tree.erasure_generation,
                purged_revisions: purged_revisions.clone(),
                recorded_at: now,
            })
            .await?;

        // Write replacement heads first, then delete the purged revisions.
        let forget_ctx = OpCtx {
            dependencies: BTreeSet::new(),
            can_pin_profile: true,
            ..ctx.clone()
        };
        for (_file, _head, draft) in new_heads {
            self.commit_change_unchecked(tx.as_mut(), &mut tree, &forget_ctx, now, draft)
                .await?;
        }
        for id in &purged_revisions {
            tx.delete_revision(id).await?;
        }
        for m in &files_purged {
            tx.purge_file(m).await?;
        }
        let mut receipts_retired = 0;
        for m in &files_affected {
            for mut r in tx.receipts_for_file(m).await? {
                if r.digest.is_some() || r.outcome.is_some() {
                    r.digest = None;
                    r.outcome = None;
                    tx.put_receipt(&r).await?;
                    receipts_retired += 1;
                }
            }
        }
        tx.put_tree(&tree).await?;
        tx.commit().await?;
        Ok(ForgetOutcome {
            erasure_seq,
            generation: tree.erasure_generation,
            files_affected,
            files_purged,
            revisions_purged: purged_revisions.len(),
            receipts_retired,
            recovery: if self.journal.independent() {
                "journaled".into()
            } else {
                "single_store".into()
            },
        })
    }

    /// Re-apply every journaled erasure, for example after restoring the
    /// ledger from a backup. If a purged revision is a file's head, the whole
    /// file is purged, because the replacement head may not exist anymore.
    pub async fn replay_erasure_journal(&self, tree_id: &str) -> Result<MaintenanceOutcome> {
        let records = self.journal.records(tree_id).await?;
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let mut out = MaintenanceOutcome::default();
        let mut touched = BTreeSet::new();
        for record in &records {
            for id in &record.purged_revisions {
                let Some(rev) = tx.revision(id).await? else {
                    continue;
                };
                let file = tx.file(&rev.memory_id).await?;
                if file
                    .as_ref()
                    .is_some_and(|f| f.head_revision_id == rev.revision_id)
                {
                    let n = tx.revisions(&rev.memory_id).await?.len();
                    tx.purge_file(&rev.memory_id).await?;
                    out.files_purged += 1;
                    out.revisions_purged += n;
                } else {
                    tx.delete_revision(id).await?;
                    out.revisions_purged += 1;
                }
                touched.insert(rev.memory_id.clone());
            }
            if record.selector == "file"
                && let Some(m) = &record.memory_id
                && tx.file(m).await?.is_some()
            {
                out.revisions_purged += tx.revisions(m).await?.len();
                tx.purge_file(m).await?;
                out.files_purged += 1;
                touched.insert(m.clone());
            }
        }
        for m in &touched {
            for mut r in tx.receipts_for_file(m).await? {
                r.digest = None;
                r.outcome = None;
                tx.put_receipt(&r).await?;
            }
        }
        if let Some(max) = records.iter().map(|r| r.generation).max() {
            tree.erasure_generation = tree.erasure_generation.max(max);
        }
        if !touched.is_empty() {
            tree.tree_revision += 1;
        }
        tx.put_tree(&tree).await?;
        tx.commit().await?;
        Ok(out)
    }

    /// Purge trash past retention and history past the revision limits.
    pub async fn sweep(&self, tree_id: &str) -> Result<MaintenanceOutcome> {
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let now = self.now();
        let mut out = MaintenanceOutcome::default();
        for file in tx.files(tree_id, true).await? {
            if self.trash_expired(&file, now) {
                out.revisions_purged += tx.revisions(&file.memory_id).await?.len();
                tx.purge_file(&file.memory_id).await?;
                for mut r in tx.receipts_for_file(&file.memory_id).await? {
                    r.digest = None;
                    r.outcome = None;
                    tx.put_receipt(&r).await?;
                }
                out.files_purged += 1;
            } else {
                out.revisions_purged += self.prune(tx.as_mut(), &file, now).await?;
            }
        }
        if out.files_purged > 0 {
            tree.tree_revision += 1;
            tx.put_tree(&tree).await?;
        }
        tx.commit().await?;
        Ok(out)
    }

    // ------------------------------------------------------------ internals

    fn trash_expired(&self, file: &FileRecord, now: DateTime<Utc>) -> bool {
        file.deleted_at
            .is_some_and(|d| now - d >= self.retention.trash)
    }

    async fn any_file(
        &self,
        tx: &mut dyn LedgerTx,
        tree_id: &str,
        memory_id: &str,
    ) -> Result<FileRecord> {
        match tx.file(memory_id).await? {
            Some(f) if f.tree_id == tree_id && !self.trash_expired(&f, self.now()) => Ok(f),
            Some(f) if f.tree_id == tree_id => Err(MemoryError::TrashExpired),
            _ => Err(MemoryError::not_found(format!("file {memory_id}"))),
        }
    }

    async fn live_file(
        &self,
        tx: &mut dyn LedgerTx,
        tree_id: &str,
        memory_id: &str,
    ) -> Result<FileRecord> {
        let f = self.any_file(tx, tree_id, memory_id).await?;
        if !f.is_live() {
            return Err(MemoryError::InTrash);
        }
        Ok(f)
    }

    /// The shared read-check-write path for single-file revisions.
    async fn mutate<F>(
        &self,
        ctx: &OpCtx,
        tree_id: &str,
        memory_id: &str,
        expected: &Head,
        request: Value,
        f: F,
    ) -> Result<WriteOutcome>
    where
        F: FnOnce(&Revision) -> Result<(FileMeta, Content, ChangeKind)> + Send,
    {
        let mut tx = self.ledger.begin().await?;
        let mut tree = load_tree(tx.as_mut(), tree_id).await?;
        let op = request["op"].as_str().unwrap_or("write").to_owned();
        let digest = match self
            .check_receipt(tx.as_mut(), ctx, tree_id, &request)
            .await?
        {
            ReceiptCheck::Replay(o) => return Ok(o),
            ReceiptCheck::Fresh(d) => d,
        };
        let file = self.live_file(tx.as_mut(), tree_id, memory_id).await?;
        check_head(&file, expected)?;
        let head = load_revision(tx.as_mut(), &file.head_revision_id).await?;
        let (meta, content, change) = f(&head)?;
        let now = self.now();
        let parents = vec![head.revision_id.clone()];
        let (file, changed) = self
            .commit_change(
                tx.as_mut(),
                &mut tree,
                ctx,
                now,
                Draft {
                    memory_id: memory_id.to_owned(),
                    current: Some(file),
                    head: Some(head),
                    meta,
                    content,
                    change,
                    parents,
                },
            )
            .await?;
        let outcome = outcome(&file, &tree, changed.is_some());
        self.store_receipt(tx.as_mut(), ctx, tree_id, &op, digest, &outcome, now)
            .await?;
        tx.commit().await?;
        Ok(outcome)
    }

    /// Validate path, permissions and limits, then write.
    async fn commit_change(
        &self,
        tx: &mut dyn LedgerTx,
        tree: &mut TreeRecord,
        ctx: &OpCtx,
        now: DateTime<Utc>,
        draft: Draft,
    ) -> Result<(FileRecord, Option<Revision>)> {
        let prior_meta = draft.head.as_ref().map(|h| h.meta.clone());
        let path_changed = prior_meta
            .as_ref()
            .is_none_or(|m| m.path != draft.meta.path);
        let live = tx.files(&tree.tree_id, false).await?;
        if path_changed {
            path::validate_path(&draft.meta.path)?;
            let key = path::collision_key(&draft.meta.path);
            if live
                .iter()
                .any(|f| f.memory_id != draft.memory_id && path::collision_key(&f.meta.path) == key)
            {
                return Err(MemoryError::PathExists(draft.meta.path.clone()));
            }
        }
        self.check_profile_permission(ctx, tree, prior_meta.as_ref(), &draft.meta)?;
        if draft.current.is_none() && live.len() >= self.limits.max_files_per_tree {
            return Err(MemoryError::Limit {
                kind: LimitKind::FilesPerTree,
                limit: self.limits.max_files_per_tree,
                actual: live.len() + 1,
            });
        }
        let other_bytes: usize = live
            .iter()
            .filter(|f| f.memory_id != draft.memory_id)
            .map(|f| f.size_bytes)
            .sum();
        let (file, rev) = self
            .commit_change_unchecked(tx, tree, ctx, now, draft)
            .await?;
        if let Some(r) = &rev {
            let total = other_bytes + r.rendered_len();
            if total > self.limits.max_rendered_bytes_per_tree {
                return Err(MemoryError::Limit {
                    kind: LimitKind::RenderedBytesPerTree,
                    limit: self.limits.max_rendered_bytes_per_tree,
                    actual: total,
                });
            }
        }
        Ok((file, rev))
    }

    /// Build the revision, check per-file limits and persist it. Returns no
    /// revision when nothing changed.
    async fn commit_change_unchecked(
        &self,
        tx: &mut dyn LedgerTx,
        tree: &mut TreeRecord,
        ctx: &OpCtx,
        now: DateTime<Utc>,
        draft: Draft,
    ) -> Result<(FileRecord, Option<Revision>)> {
        let Draft {
            memory_id,
            current,
            head,
            meta,
            content,
            change,
            parents,
        } = draft;
        let old_entries: &[Entry] = head.as_ref().map(|h| h.entries.as_slice()).unwrap_or(&[]);
        let entries = match content {
            Content::Same => old_entries.to_vec(),
            Content::Entries(e) => e,
            Content::Text(text) => self.align_text(old_entries, &text, &meta, ctx, now)?,
            Content::Realign(base, text) => self.align_text(&base, &text, &meta, ctx, now)?,
        };
        let version = version_hash(&meta, &entries);
        if let (Some(file), Some(h)) = (&current, &head)
            && h.version == version
            && h.meta == meta
        {
            return Ok((file.clone(), None));
        }
        let seq = current.as_ref().map(|f| f.head_seq + 1).unwrap_or(1);
        let revision = Revision {
            revision_id: self.ids.next("rev"),
            memory_id: memory_id.clone(),
            tree_id: tree.tree_id.clone(),
            seq,
            parents,
            change,
            actor: ctx.actor.principal.clone(),
            harness: ctx.actor.harness.clone(),
            created_at: now,
            version: version.clone(),
            meta: meta.clone(),
            entries,
        };
        self.check_file_limits(&revision)?;
        let mut dependencies = current
            .as_ref()
            .map(|f| f.dependencies.clone())
            .unwrap_or_default();
        for e in &revision.entries {
            dependencies.extend(e.lineage.dependencies.iter().cloned());
        }
        let structural = current
            .as_ref()
            .is_none_or(|f| f.meta.path != meta.path || f.meta.category_id != meta.category_id);
        let file = FileRecord {
            memory_id: memory_id.clone(),
            tree_id: tree.tree_id.clone(),
            meta,
            head_revision_id: revision.revision_id.clone(),
            head_seq: seq,
            version,
            size_bytes: revision.rendered_len(),
            entry_count: revision.entries.len(),
            created_at: current.as_ref().map(|f| f.created_at).unwrap_or(now),
            updated_at: now,
            deleted_at: current.as_ref().and_then(|f| f.deleted_at),
            dependencies,
        };
        tx.put_revision(&revision).await?;
        tx.put_file(&file).await?;
        if structural {
            tree.tree_revision += 1;
        }
        tx.put_tree(tree).await?;
        self.prune(tx, &file, now).await?;
        Ok((file, Some(revision)))
    }

    fn align_text(
        &self,
        old_entries: &[Entry],
        text: &str,
        meta: &FileMeta,
        ctx: &OpCtx,
        now: DateTime<Utc>,
    ) -> Result<Vec<Entry>> {
        codec::admit(text).map_err(|e| MemoryError::InvalidText(e.to_string()))?;
        if text.len() > self.limits.max_rendered_bytes_per_file {
            return Err(MemoryError::Limit {
                kind: LimitKind::RenderedBytesPerFile,
                limit: self.limits.max_rendered_bytes_per_file,
                actual: text.len(),
            });
        }
        let ids = self.ids.clone();
        let mut next = || ids.next("ent");
        Ok(align(
            old_entries,
            text,
            meta.entry_mode,
            AlignCtx {
                now,
                actor: &ctx.actor.principal,
                next_id: &mut next,
                dependencies: &ctx.dependencies,
            },
        ))
    }

    fn check_file_limits(&self, r: &Revision) -> Result<()> {
        let l = &self.limits;
        let checks = [
            (
                LimitKind::RenderedBytesPerFile,
                l.max_rendered_bytes_per_file,
                r.rendered_len(),
            ),
            (
                LimitKind::EntriesPerFile,
                l.max_entries_per_file,
                r.entries.len(),
            ),
            (
                LimitKind::MetadataBytesPerFile,
                l.max_metadata_bytes_per_file,
                r.metadata_bytes(),
            ),
        ];
        for (kind, limit, actual) in checks {
            if actual > limit {
                return Err(MemoryError::Limit {
                    kind,
                    limit,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn check_profile_permission(
        &self,
        ctx: &OpCtx,
        tree: &TreeRecord,
        before: Option<&FileMeta>,
        after: &FileMeta,
    ) -> Result<()> {
        let hint = |c: &Option<String>| {
            tree.categories
                .iter()
                .find(|k| k.id == *c)
                .map(|k| k.behavior_hint)
        };
        let becomes_profile = hint(&after.category_id) == Some(BehaviorHint::Profile)
            && before.is_none_or(|b| b.category_id != after.category_id);
        if becomes_profile && !ctx.can_pin_profile {
            return Err(MemoryError::PermissionDenied(
                "placing a file in a profile category needs the pin-profile permission".into(),
            ));
        }
        Ok(())
    }

    /// Drop revisions beyond the history limits. The head never counts.
    async fn prune(
        &self,
        tx: &mut dyn LedgerTx,
        file: &FileRecord,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        let revisions = tx.revisions(&file.memory_id).await?;
        let mut prior: Vec<&Revision> = revisions
            .iter()
            .filter(|r| r.revision_id != file.head_revision_id)
            .collect();
        prior.sort_by_key(|r| std::cmp::Reverse(r.seq));
        let mut purged = 0;
        for (i, r) in prior.into_iter().enumerate() {
            if i >= self.retention.max_revisions
                || now - r.created_at > self.retention.max_revision_age
            {
                tx.delete_revision(&r.revision_id).await?;
                purged += 1;
            }
        }
        Ok(purged)
    }

    async fn check_receipt(
        &self,
        tx: &mut dyn LedgerTx,
        ctx: &OpCtx,
        tree_id: &str,
        request: &Value,
    ) -> Result<ReceiptCheck> {
        let Some(key) = &ctx.idempotency_key else {
            return Ok(ReceiptCheck::Fresh(None));
        };
        let body = canonical_json(&json!({"tree": tree_id, "request": request}));
        let digest = hmac_sha256_hex(&self.receipt_key, body.as_bytes());
        match tx.receipt(tree_id, &ctx.actor.principal, key).await? {
            None => Ok(ReceiptCheck::Fresh(Some((key.clone(), digest)))),
            Some(r) => match (&r.digest, &r.outcome) {
                (Some(d), Some(o)) if *d == digest => Ok(ReceiptCheck::Replay(WriteOutcome {
                    memory_id: o.memory_id.clone().unwrap_or_default(),
                    head: Head {
                        revision_id: o.revision_id.clone().unwrap_or_default(),
                        version: o.version.clone().unwrap_or_default(),
                    },
                    seq: o.seq.unwrap_or(0),
                    tree_revision: o.tree_revision,
                    changed: false,
                    replayed: true,
                })),
                (None, _) | (_, None) => Err(MemoryError::IdempotencyKeyRetired),
                _ => Err(MemoryError::IdempotencyConflict),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn store_receipt(
        &self,
        tx: &mut dyn LedgerTx,
        ctx: &OpCtx,
        tree_id: &str,
        operation: &str,
        digest: Option<(String, String)>,
        outcome: &WriteOutcome,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let Some((key, digest)) = digest else {
            return Ok(());
        };
        tx.put_receipt(&Receipt {
            tree_id: tree_id.to_owned(),
            principal: ctx.actor.principal.clone(),
            key,
            operation: operation.to_owned(),
            digest: Some(digest),
            memory_id: Some(outcome.memory_id.clone()),
            outcome: Some(ReceiptOutcome {
                memory_id: Some(outcome.memory_id.clone()),
                revision_id: Some(outcome.head.revision_id.clone()),
                version: Some(outcome.head.version.clone()),
                seq: Some(outcome.seq),
                tree_revision: outcome.tree_revision,
            }),
            created_at: now,
        })
        .await
    }
}

struct Draft {
    memory_id: String,
    current: Option<FileRecord>,
    head: Option<Revision>,
    meta: FileMeta,
    content: Content,
    change: ChangeKind,
    parents: Vec<String>,
}

fn outcome(file: &FileRecord, tree: &TreeRecord, changed: bool) -> WriteOutcome {
    WriteOutcome {
        memory_id: file.memory_id.clone(),
        head: file.head(),
        seq: file.head_seq,
        tree_revision: tree.tree_revision,
        changed,
        replayed: false,
    }
}

fn check_head(file: &FileRecord, expected: &Head) -> Result<()> {
    if file.head() != *expected {
        return Err(MemoryError::VersionConflict {
            current: file.head(),
        });
    }
    Ok(())
}

async fn load_tree(tx: &mut dyn LedgerTx, tree_id: &str) -> Result<TreeRecord> {
    tx.tree(tree_id)
        .await?
        .ok_or_else(|| MemoryError::not_found(format!("tree {tree_id}")))
}

async fn load_revision(tx: &mut dyn LedgerTx, revision_id: &str) -> Result<Revision> {
    tx.revision(revision_id)
        .await?
        .ok_or_else(|| MemoryError::Storage(format!("revision {revision_id} is missing")))
}

/// The version hash covers visible text, file metadata and descriptive
/// annotations, so any user-visible change produces a new version.
pub fn version_hash(meta: &FileMeta, entries: &[Entry]) -> String {
    let annotations: Vec<Value> = entries
        .iter()
        .filter(|e| !e.annotation.is_empty())
        .map(|e| json!([e.entry_id, e.annotation]))
        .collect();
    let text = codec::render(entries.iter().map(|e| e.text.as_str()));
    let body = json!({"text": text, "meta": meta, "annotations": annotations});
    format!("sha256:{}", sha256_hex(canonical_json(&body).as_bytes()))
}

pub fn apply_edit(text: &str, edit: &Edit) -> Result<String> {
    match edit {
        Edit::StrReplace {
            old,
            new,
            replace_all,
        } => {
            if old.is_empty() {
                return Err(MemoryError::InvalidRequest("old text is empty".into()));
            }
            match text.matches(old.as_str()).count() {
                0 => Err(MemoryError::NoMatch),
                1 => Ok(text.replacen(old.as_str(), new, 1)),
                _ if *replace_all => Ok(text.replace(old.as_str(), new)),
                n => Err(MemoryError::MultipleMatches(n)),
            }
        }
        Edit::Insert { line, text: insert } => {
            let starts: Vec<usize> = std::iter::once(0)
                .chain(text.match_indices('\n').map(|(i, _)| i + 1))
                .filter(|i| *i < text.len() || *i == 0)
                .collect();
            let count = if text.is_empty() { 0 } else { starts.len() };
            if *line > count {
                return Err(MemoryError::InvalidRequest(format!(
                    "line {line} is past the end ({count} lines)"
                )));
            }
            let at = if *line == count {
                text.len()
            } else {
                starts[*line]
            };
            let mut out = String::with_capacity(text.len() + insert.len());
            out.push_str(&text[..at]);
            out.push_str(insert);
            out.push_str(&text[at..]);
            Ok(out)
        }
        Edit::Patch { hunks } => {
            if hunks.is_empty() {
                return Err(MemoryError::InvalidRequest("patch has no hunks".into()));
            }
            let mut out = text.to_owned();
            for h in hunks {
                out = apply_edit(
                    &out,
                    &Edit::StrReplace {
                        old: h.old.clone(),
                        new: h.new.clone(),
                        replace_all: false,
                    },
                )?;
            }
            Ok(out)
        }
        Edit::Write { text } => Ok(text.clone()),
    }
}

fn validate_display_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 120 || name.chars().any(char::is_control) {
        return Err(MemoryError::InvalidRequest(
            "display names are 1 to 120 bytes on one line".into(),
        ));
    }
    Ok(())
}

fn validate_annotation(a: &Annotation) -> Result<()> {
    if a.sources.len() > 16 || a.sources.iter().any(|s| s.len() > 512) {
        return Err(MemoryError::InvalidRequest(
            "at most 16 sources of up to 512 bytes".into(),
        ));
    }
    if a.note.as_ref().is_some_and(|n| n.len() > 1024) {
        return Err(MemoryError::InvalidRequest(
            "notes are at most 1 KiB".into(),
        ));
    }
    if let (Some(f), Some(t)) = (a.valid_from, a.valid_to)
        && t < f
    {
        return Err(MemoryError::InvalidRequest(
            "valid_to is before valid_from".into(),
        ));
    }
    Ok(())
}

/// Remove forgotten content from a head revision.
fn scrub(head: &Revision, selector: &ForgetSelector) -> (FileMeta, Content) {
    match selector {
        ForgetSelector::File { .. } => (head.meta.clone(), Content::Same),
        ForgetSelector::Entry { entry_id, .. } => {
            let entries = head
                .entries
                .iter()
                .filter(|e| &e.entry_id != entry_id)
                .cloned()
                .collect();
            (head.meta.clone(), Content::Entries(entries))
        }
        ForgetSelector::Text { text, .. } => {
            let mut meta = head.meta.clone();
            meta.description = meta.description.replace(text.as_str(), "");
            if meta.display_name.contains(text.as_str()) {
                meta.display_name = path::validate_path(&meta.path)
                    .map(|p| path::default_display_name(&p))
                    .unwrap_or_default();
            }
            // Scrub entries one by one first so identities survive, then cut
            // any occurrence that spans entry boundaries.
            let entries: Vec<Entry> = head
                .entries
                .iter()
                .cloned()
                .map(|mut e| {
                    e.text = e.text.replace(text.as_str(), "");
                    e.annotation.sources.retain(|s| !s.contains(text.as_str()));
                    if e.annotation
                        .note
                        .as_ref()
                        .is_some_and(|n| n.contains(text.as_str()))
                    {
                        e.annotation.note = None;
                    }
                    e
                })
                .filter(|e| !e.text.is_empty())
                .collect();
            let joined = codec::render(entries.iter().map(|e| e.text.as_str()));
            if joined.contains(text.as_str()) {
                let cleaned = joined.replace(text.as_str(), "");
                return (meta, Content::Realign(entries, cleaned));
            }
            (meta, Content::Entries(entries))
        }
    }
}
