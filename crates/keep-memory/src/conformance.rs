//! The behaviour suite every ledger must pass.
//!
//! Backends call [`run`] with a factory that returns a fresh, empty ledger
//! and journal for each case. The in-memory ledger runs it in this crate's
//! tests and the Postgres ledger in `keep-ledger`'s, so both are held to the
//! same semantics.

use std::future::Future;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use keep_core::memory::MemoryLimits;

use crate::codec::EntryMode;
use crate::error::{LimitKind, MemoryError};
use crate::ledger::{ErasureJournal, MemoryLedger};
use crate::model::{Annotation, ChangeKind, Confidence, Head, Revision};
use crate::service::{
    CreateFile, Edit, ForgetSelector, Hunk, ListQuery, ManualClock, MemoryService, MetadataPatch,
    OpCtx, SequentialIds,
};

pub struct Backend {
    pub ledger: Arc<dyn MemoryLedger>,
    pub journal: Arc<dyn ErasureJournal>,
}

impl Backend {
    pub fn in_memory() -> Self {
        Self {
            ledger: Arc::new(crate::inmem::InMemoryLedger::new()),
            journal: Arc::new(crate::inmem::InMemoryJournal::default()),
        }
    }
}

pub const TREE: &str = "tree_conformance";

pub fn start_time() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2030-01-01T00:00:00Z")
        .map(|t| t.with_timezone(&Utc))
        .unwrap_or_default()
}

pub struct Fixture {
    pub svc: MemoryService,
    pub clock: Arc<ManualClock>,
    pub ledger: Arc<dyn MemoryLedger>,
}

pub async fn fixture(backend: Backend, limits: MemoryLimits) -> Fixture {
    let clock = Arc::new(ManualClock::new(start_time()));
    let svc = MemoryService::new(backend.ledger.clone(), backend.journal)
        .with_clock(clock.clone())
        .with_ids(Arc::new(SequentialIds::default()))
        .with_limits(limits)
        .with_receipt_key(b"conformance-key".to_vec());
    svc.ensure_tree(TREE, "user:sam").await.expect("tree");
    Fixture {
        svc,
        clock,
        ledger: backend.ledger,
    }
}

fn owner() -> OpCtx {
    OpCtx::new("user:sam").pinning().harness("conformance")
}

macro_rules! cases {
    ($make:ident; $($case:ident),* $(,)?) => {
        $(
            eprintln!("conformance: {}", stringify!($case));
            $case(fixture($make().await, MemoryLimits::default()).await).await;
        )*
    };
}

/// Run every case. Each gets a fresh backend from `make`.
pub async fn run<F, Fut>(make: F)
where
    F: Fn() -> Fut,
    Fut: Future<Output = Backend>,
{
    cases!(make;
        lossless_create_and_view,
        compare_and_swap,
        disjoint_merge,
        text_edits,
        paths_and_renames,
        idempotent_receipts,
        history_and_undo,
        trash_lifecycle,
        lineage_is_protected,
        annotations_and_metadata,
        description_update_and_undo,
        profile_needs_permission,
        forget_text_everywhere,
        forget_entry_and_file,
    );
    eprintln!("conformance: limits");
    limits(fixture(make().await, small_limits()).await).await;
}

pub fn small_limits() -> MemoryLimits {
    MemoryLimits {
        max_rendered_bytes_per_file: 200,
        max_files_per_tree: 3,
        max_rendered_bytes_per_tree: 450,
        max_entries_per_file: 4,
        max_metadata_bytes_per_file: 64 * 1024,
    }
}

/// Every stored revision of every file in the tree, trash included.
pub async fn stored_revisions(f: &Fixture) -> Vec<Revision> {
    let mut tx = f.ledger.begin().await.expect("tx");
    let mut out = Vec::new();
    for file in tx.files(TREE, true).await.expect("files") {
        out.extend(tx.revisions(&file.memory_id).await.expect("revisions"));
    }
    out
}

async fn lossless_create_and_view(f: Fixture) {
    let text = "---\ntitle: x\n---\nLead  \r\n\n# Prefs\n- tea\n- **no** coffee\n\n```\n# not a heading\n```\n## Deep\ntail without newline";
    let out = f
        .svc
        .create(&owner(), TREE, &CreateFile::new("/topics/drinks.md", text))
        .await
        .expect("create");
    assert!(out.changed && !out.replayed);
    assert_eq!(out.seq, 1);
    let view = f.svc.view(TREE, &out.memory_id).await.expect("view");
    assert_eq!(view.text(), text);
    assert_eq!(view.file.meta.category_id.as_deref(), Some("topics"));
    assert_eq!(view.file.meta.display_name, "Drinks");
    assert_eq!(view.file.size_bytes, text.len());
    // front matter, lead, # Prefs, ## Deep
    assert_eq!(view.revision.entries.len(), 4);
    let by_path = f
        .svc
        .view_by_path(TREE, "/topics/drinks.md")
        .await
        .expect("by path");
    assert_eq!(by_path.revision, view.revision);

    let bullets = f
        .svc
        .create(
            &owner(),
            TREE,
            &CreateFile::new("/topics/list.md", "# L\n- a\n- b\n")
                .mode(EntryMode::SectionsAndBullets),
        )
        .await
        .expect("bullets");
    let v = f.svc.view(TREE, &bullets.memory_id).await.expect("view");
    assert_eq!(v.revision.entries.len(), 3);
    assert_eq!(v.text(), "# L\n- a\n- b\n");

    let page = f
        .svc
        .list(
            TREE,
            &ListQuery {
                limit: 1,
                ..ListQuery::default()
            },
        )
        .await
        .expect("list");
    assert_eq!(page.total, 2);
    assert_eq!(page.files[0].meta.path, "/topics/drinks.md");
    let page2 = f
        .svc
        .list(
            TREE,
            &ListQuery {
                limit: 1,
                after: page.next_after.clone(),
                ..ListQuery::default()
            },
        )
        .await
        .expect("list 2");
    assert_eq!(page2.files[0].meta.path, "/topics/list.md");
    assert!(page2.next_after.is_none());

    let bad = f
        .svc
        .create(&owner(), TREE, &CreateFile::new("/topics/zero.md", "a\0b"))
        .await;
    assert!(matches!(bad, Err(MemoryError::InvalidText(_))));
}

async fn compare_and_swap(f: Fixture) {
    let ctx = owner();
    let a = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/cas.md", "A\n"))
        .await
        .expect("create");
    let write = |t: &str| Edit::Write { text: t.into() };
    let b = f
        .svc
        .edit(&ctx, TREE, &a.memory_id, &a.head, &write("B\n"))
        .await
        .expect("edit");
    let stale = f
        .svc
        .edit(&ctx, TREE, &a.memory_id, &a.head, &write("C\n"))
        .await;
    match stale {
        Err(MemoryError::VersionConflict { current }) => assert_eq!(current, b.head),
        other => panic!("expected conflict, got {other:?}"),
    }
    // A -> B -> undo gives A's text and version under a new revision id; a
    // client still holding A's head must not win.
    let undone = f
        .svc
        .undo(&ctx, TREE, &a.memory_id, &b.head, None)
        .await
        .expect("undo");
    assert_eq!(undone.head.version, a.head.version);
    assert_ne!(undone.head.revision_id, a.head.revision_id);
    assert_eq!(undone.seq, 3);
    let aba = f
        .svc
        .edit(&ctx, TREE, &a.memory_id, &a.head, &write("D\n"))
        .await;
    assert!(matches!(aba, Err(MemoryError::VersionConflict { .. })));
    // A no-op write does not create a revision.
    let same = f
        .svc
        .edit(&ctx, TREE, &a.memory_id, &undone.head, &write("A\n"))
        .await
        .expect("noop");
    assert!(!same.changed);
    assert_eq!(same.head, undone.head);
    // Tree-level CAS on create.
    let tree = f.svc.tree(TREE).await.expect("tree");
    let mut req = CreateFile::new("/topics/new.md", "x");
    req.expected_tree_revision = Some(tree.tree_revision + 5);
    assert!(matches!(
        f.svc.create(&ctx, TREE, &req).await,
        Err(MemoryError::TreeConflict { .. })
    ));
    req.expected_tree_revision = Some(tree.tree_revision);
    f.svc
        .create(&ctx, TREE, &req)
        .await
        .expect("create at tree rev");
}

async fn disjoint_merge(f: Fixture) {
    let ctx = owner();
    let base = "# Garden\nTomatoes in bed one.\n\n# Garage\nKayak on the left wall.\n";
    let a = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/house.md", base))
        .await
        .expect("create");
    let replace = |old: &str, new: &str| Edit::StrReplace {
        old: old.into(),
        new: new.into(),
        replace_all: false,
    };
    let theirs = f
        .svc
        .edit(
            &ctx,
            TREE,
            &a.memory_id,
            &a.head,
            &replace("bed one", "bed two"),
        )
        .await
        .expect("their edit");
    // Plain edit keeps strict compare-and-swap.
    let edit = replace("left wall", "right wall");
    assert!(matches!(
        f.svc.edit(&ctx, TREE, &a.memory_id, &a.head, &edit).await,
        Err(MemoryError::VersionConflict { .. })
    ));
    let keyed = owner().with_key("merge-1");
    let merged = f
        .svc
        .edit_merging(&keyed, TREE, &a.memory_id, &a.head, &edit)
        .await
        .expect("disjoint edits merge");
    assert!(merged.changed && merged.seq == theirs.seq + 1);
    let view = f.svc.view(TREE, &a.memory_id).await.expect("view");
    assert_eq!(
        view.text(),
        "# Garden\nTomatoes in bed two.\n\n# Garage\nKayak on the right wall.\n"
    );
    assert_eq!(view.revision.change, ChangeKind::Merge);
    let again = f
        .svc
        .edit_merging(&keyed, TREE, &a.memory_id, &a.head, &edit)
        .await
        .expect("replay");
    assert!(again.replayed && again.head == merged.head);
    // Overlapping changes are refused and nothing is written.
    let clash = f
        .svc
        .edit_merging(
            &ctx,
            TREE,
            &a.memory_id,
            &a.head,
            &replace("bed one", "bed three"),
        )
        .await;
    assert!(
        matches!(clash, Err(MemoryError::MergeConflict { .. })),
        "{clash:?}"
    );
    let after = f.svc.view(TREE, &a.memory_id).await.expect("view");
    assert_eq!(after.file.head(), merged.head);
}

async fn text_edits(f: Fixture) {
    let ctx = owner();
    let c = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/e.md", "one\ntwo\ntwo\n"),
        )
        .await
        .expect("create");
    let mut head = c.head;
    let id = c.memory_id;
    let replace = |old: &str, new: &str, all| Edit::StrReplace {
        old: old.into(),
        new: new.into(),
        replace_all: all,
    };
    let insert = |line, text: &str| Edit::Insert {
        line,
        text: text.into(),
    };
    let hunk = |old: &str, new: &str| Hunk {
        old: old.into(),
        new: new.into(),
    };
    assert_eq!(
        f.svc
            .edit(&ctx, TREE, &id, &head, &replace("zzz", "y", false))
            .await,
        Err(MemoryError::NoMatch)
    );
    assert_eq!(
        f.svc
            .edit(&ctx, TREE, &id, &head, &replace("two", "2", false))
            .await,
        Err(MemoryError::MultipleMatches(2))
    );
    head = f
        .svc
        .edit(&ctx, TREE, &id, &head, &replace("two", "2", true))
        .await
        .expect("replace all")
        .head;
    head = f
        .svc
        .edit(&ctx, TREE, &id, &head, &insert(1, "ins\n"))
        .await
        .expect("insert")
        .head;
    head = f
        .svc
        .edit(&ctx, TREE, &id, &head, &insert(4, "end\n"))
        .await
        .expect("append")
        .head;
    assert!(matches!(
        f.svc.edit(&ctx, TREE, &id, &head, &insert(9, "x")).await,
        Err(MemoryError::InvalidRequest(_))
    ));
    let patch = Edit::Patch {
        hunks: vec![hunk("one", "uno"), hunk("end", "fin")],
    };
    head = f
        .svc
        .edit(&ctx, TREE, &id, &head, &patch)
        .await
        .expect("patch")
        .head;
    // A failing hunk applies nothing.
    let bad = Edit::Patch {
        hunks: vec![hunk("uno", "1"), hunk("missing", "x")],
    };
    assert_eq!(
        f.svc.edit(&ctx, TREE, &id, &head, &bad).await,
        Err(MemoryError::NoMatch)
    );
    let view = f.svc.view(TREE, &id).await.expect("view");
    assert_eq!(view.text(), "uno\nins\n2\n2\nfin\n");
    assert_eq!(view.head(), head);
}

async fn paths_and_renames(f: Fixture) {
    let ctx = owner();
    let a = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/alpha.md", "a"))
        .await
        .expect("a");
    f.svc
        .create(&ctx, TREE, &CreateFile::new("/topics/beta.md", "b"))
        .await
        .expect("b");
    for clash in ["/topics/alpha.md", "/Topics/ALPHA.md"] {
        assert!(matches!(
            f.svc.create(&ctx, TREE, &CreateFile::new(clash, "x")).await,
            Err(MemoryError::PathExists(_))
        ));
    }
    assert!(matches!(
        f.svc
            .create(&ctx, TREE, &CreateFile::new("/topics/con.md", "x"))
            .await,
        Err(MemoryError::InvalidPath(_))
    ));
    assert!(matches!(
        f.svc
            .rename(&ctx, TREE, &a.memory_id, &a.head, "/topics/Beta.md")
            .await,
        Err(MemoryError::PathExists(_))
    ));
    let before = f.svc.tree(TREE).await.expect("tree").tree_revision;
    let project = "6f1c2a52-3b6e-4b0e-9a51-1f3f7e8d2c10";
    let moved = f
        .svc
        .rename(
            &ctx,
            TREE,
            &a.memory_id,
            &a.head,
            &format!("/projects/{project}/index.md"),
        )
        .await
        .expect("rename");
    assert!(moved.tree_revision > before);
    let v = f.svc.view(TREE, &a.memory_id).await.expect("view");
    assert_eq!(v.file.meta.category_id.as_deref(), Some("projects"));
    assert_eq!(v.file.meta.project_id.as_deref(), Some(project));
    assert_eq!(v.file.meta.display_name, "Index");
    assert_eq!(v.text(), "a");
    // The old path is free again.
    f.svc
        .create(&ctx, TREE, &CreateFile::new("/topics/alpha.md", "again"))
        .await
        .expect("reuse path");
}

async fn idempotent_receipts(f: Fixture) {
    let ctx = owner().with_key("req-1");
    let req = CreateFile::new("/topics/idem.md", "once");
    let first = f.svc.create(&ctx, TREE, &req).await.expect("first");
    let again = f.svc.create(&ctx, TREE, &req).await.expect("replay");
    assert!(again.replayed);
    assert_eq!(again.head, first.head);
    assert_eq!(again.memory_id, first.memory_id);
    let page = f.svc.list(TREE, &ListQuery::default()).await.expect("list");
    assert_eq!(page.total, 1);
    assert_eq!(
        f.svc
            .create(&ctx, TREE, &CreateFile::new("/topics/other.md", "x"))
            .await,
        Err(MemoryError::IdempotencyConflict)
    );
    // Keys are scoped per principal.
    let other = OpCtx::new("user:kim").with_key("req-1");
    f.svc
        .create(&other, TREE, &CreateFile::new("/topics/kim.md", "x"))
        .await
        .expect("other principal");
    // Receipts never store request text.
    let mut tx = f.ledger.begin().await.expect("tx");
    let receipts = tx
        .receipts_for_file(&first.memory_id)
        .await
        .expect("receipts");
    assert_eq!(receipts.len(), 1);
    for r in receipts {
        let json = serde_json::to_string(&r).expect("json");
        assert!(!json.contains("once"));
    }
}

async fn history_and_undo(f: Fixture) {
    let ctx = owner();
    let c = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/h.md", "v0\n"))
        .await
        .expect("create");
    let id = c.memory_id.clone();
    let mut head = c.head.clone();
    for i in 1..=120 {
        f.clock.advance(Duration::seconds(1));
        head = f
            .svc
            .edit(
                &ctx,
                TREE,
                &id,
                &head,
                &Edit::Write {
                    text: format!("v{i}\n"),
                },
            )
            .await
            .expect("edit")
            .head;
    }
    let history = f.svc.history(TREE, &id).await.expect("history");
    assert_eq!(history.len(), 101, "100 prior revisions plus the head");
    assert!(history[0].is_head);
    assert_eq!(history[0].seq, 121);
    assert_eq!(history[100].seq, 21);
    assert!(matches!(
        f.svc.view_revision(TREE, &id, &c.head.revision_id).await,
        Err(MemoryError::RevisionNotRetained(_))
    ));
    assert!(matches!(
        f.svc
            .undo(&ctx, TREE, &id, &head, Some(&c.head.revision_id))
            .await,
        Err(MemoryError::RevisionNotRetained(_))
    ));
    // Undo to an explicit retained revision restores its text as a new
    // revision.
    let target = history[50].clone();
    let undone = f
        .svc
        .undo(&ctx, TREE, &id, &head, Some(&target.revision_id))
        .await
        .expect("undo");
    assert_eq!(undone.seq, 122);
    let view = f.svc.view(TREE, &id).await.expect("view");
    assert_eq!(view.text(), format!("v{}\n", target.seq - 1));
    assert_eq!(view.revision.change, ChangeKind::Undo);
    // After 31 days only the head survives the next save's pruning.
    f.clock.advance(Duration::days(31));
    let next = f
        .svc
        .edit(
            &ctx,
            TREE,
            &id,
            &undone.head,
            &Edit::Write {
                text: "late\n".into(),
            },
        )
        .await
        .expect("late edit");
    let history = f.svc.history(TREE, &id).await.expect("history");
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].revision_id, next.head.revision_id);
    // A fresh file with a single revision has nothing to undo.
    let lone = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/lone.md", "x"))
        .await
        .expect("lone");
    assert_eq!(
        f.svc
            .undo(&ctx, TREE, &lone.memory_id, &lone.head, None)
            .await,
        Err(MemoryError::NothingToUndo)
    );
}

async fn trash_lifecycle(f: Fixture) {
    let ctx = owner();
    let a = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/t.md", "keep me"))
        .await
        .expect("create");
    let bogus = Head {
        revision_id: "rev_x".into(),
        version: "v".into(),
    };
    assert!(matches!(
        f.svc.delete(&ctx, TREE, &a.memory_id, &bogus).await,
        Err(MemoryError::VersionConflict { .. })
    ));
    f.svc
        .delete(&ctx, TREE, &a.memory_id, &a.head)
        .await
        .expect("delete");
    assert_eq!(
        f.svc.view(TREE, &a.memory_id).await,
        Err(MemoryError::InTrash)
    );
    let page = f.svc.list(TREE, &ListQuery::default()).await.expect("list");
    assert_eq!(page.total, 0);
    let trash_query = ListQuery {
        include_trashed: true,
        ..ListQuery::default()
    };
    let trash = f.svc.list(TREE, &trash_query).await.expect("trash");
    assert_eq!(trash.total, 1);
    // The path is free while the file sits in the trash.
    let b = f
        .svc
        .create(&ctx, TREE, &CreateFile::new("/topics/t.md", "replacement"))
        .await
        .expect("reuse");
    assert!(matches!(
        f.svc.restore(&ctx, TREE, &a.memory_id).await,
        Err(MemoryError::PathExists(_))
    ));
    f.svc
        .delete(&ctx, TREE, &b.memory_id, &b.head)
        .await
        .expect("delete b");
    f.clock.advance(Duration::days(6));
    f.svc
        .restore(&ctx, TREE, &a.memory_id)
        .await
        .expect("restore");
    assert_eq!(
        f.svc.view(TREE, &a.memory_id).await.expect("view").text(),
        "keep me"
    );
    assert_eq!(
        f.svc.restore(&ctx, TREE, &a.memory_id).await,
        Err(MemoryError::NotInTrash)
    );
    // b expires after 7 days in the trash.
    f.clock.advance(Duration::days(1));
    assert_eq!(
        f.svc.restore(&ctx, TREE, &b.memory_id).await,
        Err(MemoryError::TrashExpired)
    );
    assert_eq!(
        f.svc.list(TREE, &trash_query).await.expect("trash").total,
        1,
        "expired files leave the trash listing"
    );
    let swept = f.svc.sweep(TREE).await.expect("sweep");
    assert_eq!(swept.files_purged, 1);
    assert!(
        stored_revisions(&f)
            .await
            .iter()
            .all(|r| r.memory_id != b.memory_id)
    );
    assert!(matches!(
        f.svc.restore(&ctx, TREE, &b.memory_id).await,
        Err(MemoryError::NotFound { .. })
    ));
}

async fn lineage_is_protected(f: Fixture) {
    let restricted = owner().depends_on("source:private-chat");
    let c = f
        .svc
        .create(
            &restricted,
            TREE,
            &CreateFile::new("/topics/l.md", "# Secret\nfrom a private chat\n"),
        )
        .await
        .expect("create");
    // A plain caller rewrites everything, including fake front matter that
    // tries to declare metadata. Lineage must survive.
    let plain = owner();
    let forged = "---\ndependencies: []\nentry_id: forged\n---\nunrelated text\n";
    let rewritten = f
        .svc
        .edit(
            &plain,
            TREE,
            &c.memory_id,
            &c.head,
            &Edit::Write {
                text: forged.into(),
            },
        )
        .await
        .expect("rewrite");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert_eq!(v.text(), forged);
    assert!(v.file.dependencies.contains("source:private-chat"));
    assert!(
        v.revision
            .entries
            .iter()
            .all(|e| e.lineage.dependencies.contains("source:private-chat"))
    );
    assert!(v.revision.entries.iter().all(|e| e.entry_id != "forged"));
    // Even shrinking to nothing keeps the file-level record.
    f.svc
        .edit(
            &plain,
            TREE,
            &c.memory_id,
            &rewritten.head,
            &Edit::Write {
                text: String::new(),
            },
        )
        .await
        .expect("empty");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert!(v.file.dependencies.contains("source:private-chat"));
    assert_eq!(v.text(), "");
}

async fn annotations_and_metadata(f: Fixture) {
    let ctx = owner();
    let c = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/a.md", "# One\nx\n# Two\ny\n"),
        )
        .await
        .expect("create");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    let entry = v.revision.entries[1].entry_id.clone();
    let annotation = Annotation {
        sources: vec!["turn:42".into()],
        valid_from: Some(start_time()),
        valid_to: None,
        confidence: Some(Confidence::High),
        note: Some("said twice".into()),
    };
    let a = f
        .svc
        .annotate(&ctx, TREE, &c.memory_id, &c.head, &entry, &annotation)
        .await
        .expect("annotate");
    assert_ne!(
        a.head.version, c.head.version,
        "annotations change the version"
    );
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert_eq!(v.text(), "# One\nx\n# Two\ny\n");
    assert_eq!(v.revision.entries[1].annotation, annotation);
    // Editing the section's text keeps the entry and its annotation.
    let e = f
        .svc
        .edit(
            &ctx,
            TREE,
            &c.memory_id,
            &a.head,
            &Edit::StrReplace {
                old: "y\n".into(),
                new: "y, revised\n".into(),
                replace_all: false,
            },
        )
        .await
        .expect("edit");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert_eq!(v.revision.entries[1].entry_id, entry);
    assert_eq!(v.revision.entries[1].annotation, annotation);
    let bad = Annotation {
        valid_from: Some(start_time()),
        valid_to: Some(start_time() - Duration::days(1)),
        ..Annotation::default()
    };
    assert!(matches!(
        f.svc
            .annotate(&ctx, TREE, &c.memory_id, &e.head, &entry, &bad)
            .await,
        Err(MemoryError::InvalidRequest(_))
    ));
    let m = f
        .svc
        .update_metadata(
            &ctx,
            TREE,
            &c.memory_id,
            &e.head,
            &MetadataPatch {
                display_name: Some("Alpha".into()),
                description: Some("Two sections".into()),
                entry_mode: Some(EntryMode::Raw),
            },
        )
        .await
        .expect("metadata");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert_eq!(v.file.meta.display_name, "Alpha");
    assert_eq!(v.file.meta.description, "Two sections");
    assert_eq!(v.revision.entries.len(), 1);
    assert_eq!(v.text(), "# One\nx\n# Two\ny, revised\n");
    let two_lines = MetadataPatch {
        description: Some("two\nlines".into()),
        ..MetadataPatch::default()
    };
    assert!(matches!(
        f.svc
            .update_metadata(&ctx, TREE, &c.memory_id, &m.head, &two_lines)
            .await,
        Err(MemoryError::InvalidRequest(_))
    ));
}

/// A description change is a compare-and-swap revision like any other: it
/// shows in history, leaves the text alone and can be undone.
async fn description_update_and_undo(f: Fixture) {
    let ctx = owner();
    let c = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/d.md", "# Notes\nkeep me\n").describe("First"),
        )
        .await
        .expect("create");
    let id = c.memory_id.clone();
    let patch = MetadataPatch {
        description: Some("Second".into()),
        ..MetadataPatch::default()
    };
    let stale = Head {
        revision_id: c.head.revision_id.clone(),
        version: "stale".into(),
    };
    assert!(matches!(
        f.svc.update_metadata(&ctx, TREE, &id, &stale, &patch).await,
        Err(MemoryError::VersionConflict { .. })
    ));
    f.clock.advance(Duration::seconds(1));
    let m = f
        .svc
        .update_metadata(&ctx, TREE, &id, &c.head, &patch)
        .await
        .expect("metadata");
    assert!(m.changed);
    assert_ne!(m.head.version, c.head.version);
    let v = f.svc.view(TREE, &id).await.expect("view");
    assert_eq!(v.file.meta.description, "Second");
    assert_eq!(v.text(), "# Notes\nkeep me\n");
    assert_eq!(v.revision.change, ChangeKind::Metadata);
    // The old head no longer matches.
    assert!(matches!(
        f.svc
            .update_metadata(&ctx, TREE, &id, &c.head, &patch)
            .await,
        Err(MemoryError::VersionConflict { .. })
    ));
    let history = f.svc.history(TREE, &id).await.expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].change, ChangeKind::Metadata);
    assert_eq!(history[0].actor, "user:sam");
    f.clock.advance(Duration::seconds(1));
    let undone = f
        .svc
        .undo(&ctx, TREE, &id, &m.head, None)
        .await
        .expect("undo");
    let v = f.svc.view(TREE, &id).await.expect("view");
    assert_eq!(v.file.meta.description, "First");
    assert_eq!(v.text(), "# Notes\nkeep me\n");
    assert_eq!(undone.head.version, c.head.version);
    assert_eq!(f.svc.history(TREE, &id).await.expect("history").len(), 3);
}

async fn profile_needs_permission(f: Fixture) {
    let plain = OpCtx::new("agent:helper");
    assert!(matches!(
        f.svc
            .create(&plain, TREE, &CreateFile::new("/profile.md", "I like tea"))
            .await,
        Err(MemoryError::PermissionDenied(_))
    ));
    let t = f
        .svc
        .create(&plain, TREE, &CreateFile::new("/topics/tea.md", "tea"))
        .await
        .expect("topic");
    assert!(matches!(
        f.svc
            .rename(&plain, TREE, &t.memory_id, &t.head, "/profile.md")
            .await,
        Err(MemoryError::PermissionDenied(_))
    ));
    let p = f
        .svc
        .create(
            &owner(),
            TREE,
            &CreateFile::new("/profile.md", "I like tea"),
        )
        .await
        .expect("pinned");
    // Editing an existing profile file needs no extra permission.
    f.svc
        .edit(
            &plain,
            TREE,
            &p.memory_id,
            &p.head,
            &Edit::Write {
                text: "I like green tea".into(),
            },
        )
        .await
        .expect("edit profile");
}

const SECRET: &str = "Rosewater-4417";

fn secret_notes() -> CreateFile {
    CreateFile::new(
        "/topics/notes.md",
        "# Codes\nThe door code is Rosewater-4417.\n# Other\nkeep this\n",
    )
    .describe("Has Rosewater-4417 inside")
}

async fn forget_text_everywhere(f: Fixture) {
    let ctx = owner();
    let key_ctx = owner().with_key("make-notes");
    let notes = f
        .svc
        .create(&key_ctx, TREE, &secret_notes())
        .await
        .expect("notes");
    let edited = f
        .svc
        .edit(
            &ctx,
            TREE,
            &notes.memory_id,
            &notes.head,
            &Edit::StrReplace {
                old: "keep this".into(),
                new: "keep this too".into(),
                replace_all: false,
            },
        )
        .await
        .expect("edit");
    let v = f.svc.view(TREE, &notes.memory_id).await.expect("view");
    let other_entry = v.revision.entries[1].entry_id.clone();
    let tagged = Annotation {
        note: Some(format!("mentions {SECRET}")),
        sources: vec![format!("turn:{SECRET}"), "turn:9".into()],
        ..Annotation::default()
    };
    let annotated = f
        .svc
        .annotate(
            &ctx,
            TREE,
            &notes.memory_id,
            &edited.head,
            &other_entry,
            &tagged,
        )
        .await
        .expect("annotate");
    // An older revision held the text, then a later one removed it.
    let old = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/old.md", format!("{SECRET} once\n")),
        )
        .await
        .expect("old");
    f.svc
        .edit(
            &ctx,
            TREE,
            &old.memory_id,
            &old.head,
            &Edit::Write {
                text: "nothing here\n".into(),
            },
        )
        .await
        .expect("edit old");
    // A trashed file also holds it.
    let trashed = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/bin.md", format!("bin: {SECRET}\n")),
        )
        .await
        .expect("trashed");
    f.svc
        .delete(&ctx, TREE, &trashed.memory_id, &trashed.head)
        .await
        .expect("delete");
    let clean = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/clean.md", "unrelated\n"),
        )
        .await
        .expect("clean");

    let before_gen = f.svc.tree(TREE).await.expect("tree").erasure_generation;
    let out = f
        .svc
        .forget(
            &ctx,
            TREE,
            &ForgetSelector::Text {
                memory_id: None,
                text: SECRET.into(),
            },
        )
        .await
        .expect("forget");
    assert_eq!(out.generation, before_gen + 1);
    assert_eq!(out.files_affected.len(), 3);
    assert!(!out.files_affected.contains(&clean.memory_id));
    assert!(out.receipts_retired >= 1);

    for r in stored_revisions(&f).await {
        let json = serde_json::to_string(&r).expect("json");
        assert!(
            !json.contains(SECRET),
            "revision {} still holds it",
            r.revision_id
        );
    }
    let v = f.svc.view(TREE, &notes.memory_id).await.expect("view");
    assert_eq!(
        v.text(),
        "# Codes\nThe door code is .\n# Other\nkeep this too\n"
    );
    assert!(!v.file.meta.description.contains(SECRET));
    assert_eq!(
        v.revision.entries[1].annotation.sources,
        vec!["turn:9".to_string()]
    );
    assert_eq!(v.revision.entries[1].annotation.note, None);
    assert_eq!(
        v.revision.entries[1].entry_id, other_entry,
        "identities survive"
    );
    // The trashed copy is scrubbed and still in the trash.
    let trash = f
        .svc
        .list(
            TREE,
            &ListQuery {
                include_trashed: true,
                ..ListQuery::default()
            },
        )
        .await
        .expect("list");
    assert!(
        trash
            .files
            .iter()
            .any(|f| f.memory_id == trashed.memory_id && !f.is_live())
    );
    for file in &trash.files {
        assert!(!serde_json::to_string(file).expect("json").contains(SECRET));
    }
    // No history path leads back to the text.
    assert_eq!(
        f.svc
            .undo(&ctx, TREE, &notes.memory_id, &v.head(), None)
            .await,
        Err(MemoryError::NothingToUndo)
    );
    for rev in [&notes.head, &edited.head, &annotated.head] {
        assert!(matches!(
            f.svc
                .view_revision(TREE, &notes.memory_id, &rev.revision_id)
                .await,
            Err(MemoryError::RevisionNotRetained(_))
        ));
    }
    // The receipt that created the file can no longer replay it.
    assert_eq!(
        f.svc.create(&key_ctx, TREE, &secret_notes()).await,
        Err(MemoryError::IdempotencyKeyRetired)
    );
    let mut tx = f.ledger.begin().await.expect("tx");
    let receipts = tx
        .receipts_for_file(&notes.memory_id)
        .await
        .expect("receipts");
    assert!(!receipts.is_empty());
    for r in receipts {
        assert!(r.digest.is_none() && r.outcome.is_none());
    }
}

async fn forget_entry_and_file(f: Fixture) {
    let ctx = owner();
    let c = f
        .svc
        .create(
            &ctx,
            TREE,
            &CreateFile::new("/topics/f.md", "# Keep\nok\n# Drop\nsensitive\n"),
        )
        .await
        .expect("create");
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    let drop_id = v.revision.entries[1].entry_id.clone();
    let replace = |old: &str, new: &str| Edit::StrReplace {
        old: old.into(),
        new: new.into(),
        replace_all: false,
    };
    let e = f
        .svc
        .edit(
            &ctx,
            TREE,
            &c.memory_id,
            &c.head,
            &replace("sensitive", "sensitive, edited"),
        )
        .await
        .expect("edit");
    let e2 = f
        .svc
        .edit(&ctx, TREE, &c.memory_id, &e.head, &replace("ok", "ok!"))
        .await
        .expect("edit 2");
    let out = f
        .svc
        .forget(
            &ctx,
            TREE,
            &ForgetSelector::Entry {
                memory_id: c.memory_id.clone(),
                entry_id: drop_id.clone(),
            },
        )
        .await
        .expect("forget entry");
    assert_eq!(out.revisions_purged, 3);
    let v = f.svc.view(TREE, &c.memory_id).await.expect("view");
    assert_eq!(v.text(), "# Keep\nok!\n");
    for r in stored_revisions(&f).await {
        assert!(r.entries.iter().all(|e| e.entry_id != drop_id));
        assert!(!r.text().contains("sensitive"));
    }
    assert!(matches!(
        f.svc
            .view_revision(TREE, &c.memory_id, &e2.head.revision_id)
            .await,
        Err(MemoryError::RevisionNotRetained(_))
    ));

    let out = f
        .svc
        .forget(
            &ctx,
            TREE,
            &ForgetSelector::File {
                memory_id: c.memory_id.clone(),
            },
        )
        .await
        .expect("forget file");
    assert_eq!(out.files_purged, vec![c.memory_id.clone()]);
    assert!(matches!(
        f.svc.view(TREE, &c.memory_id).await,
        Err(MemoryError::NotFound { .. })
    ));
    assert!(stored_revisions(&f).await.is_empty());
    // Forgetting text that does not exist is reported, not silently ignored.
    assert!(matches!(
        f.svc
            .forget(
                &ctx,
                TREE,
                &ForgetSelector::Text {
                    memory_id: None,
                    text: "absent".into()
                }
            )
            .await,
        Err(MemoryError::NotFound { .. })
    ));
}

async fn limits(f: Fixture) {
    let ctx = owner();
    let create = |path: &str, text: String| CreateFile::new(path, text);
    assert!(matches!(
        f.svc
            .create(&ctx, TREE, &create("/topics/big.md", "x".repeat(201)))
            .await,
        Err(MemoryError::Limit {
            kind: LimitKind::RenderedBytesPerFile,
            limit: 200,
            actual: 201
        })
    ));
    assert!(matches!(
        f.svc
            .create(
                &ctx,
                TREE,
                &create("/topics/many.md", "# a\n# b\n# c\n# d\n# e\n".into())
            )
            .await,
        Err(MemoryError::Limit {
            kind: LimitKind::EntriesPerFile,
            ..
        })
    ));
    let a = f
        .svc
        .create(&ctx, TREE, &create("/topics/a.md", "a".repeat(200)))
        .await
        .expect("a");
    f.svc
        .create(&ctx, TREE, &create("/topics/b.md", "b".repeat(200)))
        .await
        .expect("b");
    assert!(matches!(
        f.svc
            .create(&ctx, TREE, &create("/topics/c.md", "c".repeat(60)))
            .await,
        Err(MemoryError::Limit {
            kind: LimitKind::RenderedBytesPerTree,
            limit: 450,
            actual: 460
        })
    ));
    let c = f
        .svc
        .create(&ctx, TREE, &create("/topics/c.md", "c".into()))
        .await
        .expect("c");
    assert!(matches!(
        f.svc
            .create(&ctx, TREE, &create("/topics/d.md", "d".into()))
            .await,
        Err(MemoryError::Limit {
            kind: LimitKind::FilesPerTree,
            limit: 3,
            actual: 4
        })
    ));
    // A rejected edit leaves the file unchanged.
    assert!(matches!(
        f.svc
            .edit(
                &ctx,
                TREE,
                &c.memory_id,
                &c.head,
                &Edit::Write {
                    text: "c".repeat(100)
                }
            )
            .await,
        Err(MemoryError::Limit {
            kind: LimitKind::RenderedBytesPerTree,
            ..
        })
    ));
    assert_eq!(
        f.svc.view(TREE, &c.memory_id).await.expect("view").text(),
        "c"
    );
    // Trashed files free their quota; restoring checks it again.
    f.svc
        .delete(&ctx, TREE, &a.memory_id, &a.head)
        .await
        .expect("delete");
    f.svc
        .create(&ctx, TREE, &create("/topics/d.md", "d".repeat(200)))
        .await
        .expect("d");
    assert!(matches!(
        f.svc.restore(&ctx, TREE, &a.memory_id).await,
        Err(MemoryError::Limit {
            kind: LimitKind::FilesPerTree,
            ..
        })
    ));
    let desc = create("/topics/e.md", "e".into()).describe("d".repeat(161));
    assert!(matches!(
        f.svc.create(&ctx, TREE, &desc).await,
        Err(MemoryError::Limit {
            kind: LimitKind::DescriptionBytes,
            ..
        })
    ));
}
