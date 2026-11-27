//! The graph the dev view and the site demo draw: Knowledge entities,
//! facts, sources and collections, and the Memory tree, as one list of
//! nodes and edges.

use std::collections::{BTreeMap, BTreeSet};

use keep_core::memory::Category;
use keep_knowledge::GraphSlice;
use keep_knowledge::model::FactStatus;
use keep_memory::service::FileView;
use serde::Serialize;

/// Shortest entity name a Memory file's text is matched against, so short
/// names such as "Al" do not link every file.
const MIN_MENTION_CHARS: usize = 4;

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Graph {
    pub scope: String,
    pub truncated: bool,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Serialize)]
pub struct Node {
    pub id: String,
    pub kind: &'static str,
    pub label: String,
    /// Entity kind, a source's collection or a file's category.
    pub group: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub data: BTreeMap<&'static str, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub label: String,
}

/// The Memory side of a scope: its categories and live files.
pub struct MemoryTree<'a> {
    pub categories: &'a [Category],
    pub files: &'a [FileView],
}

pub fn build(scope: String, slice: &GraphSlice, memory: MemoryTree<'_>) -> Graph {
    let mut g = Graph {
        scope,
        truncated: slice.truncated,
        ..Graph::default()
    };
    let collections: BTreeSet<&str> = slice
        .sources
        .iter()
        .map(|s| s.collection.as_str())
        .collect();
    for c in collections {
        g.nodes.push(Node {
            id: collection_id(c),
            kind: "collection",
            label: c.to_owned(),
            group: c.to_owned(),
            data: BTreeMap::new(),
        });
    }
    for s in &slice.sources {
        let mut data = BTreeMap::new();
        data.insert("sourceKind", s.kind.as_str().into());
        if let Some(url) = s.metadata.get("url") {
            data.insert("url", url.as_str().into());
        }
        g.nodes.push(Node {
            id: s.id.clone(),
            kind: "source",
            label: s.title.clone(),
            group: s.collection.clone(),
            data,
        });
        g.edges.push(Edge {
            id: format!("{}>{}", s.id, collection_id(&s.collection)),
            source: s.id.clone(),
            target: collection_id(&s.collection),
            kind: "in_collection",
            label: String::new(),
        });
    }
    let sources: BTreeSet<&str> = slice.sources.iter().map(|s| s.id.as_str()).collect();
    let entities: BTreeSet<&str> = slice.entities.iter().map(|e| e.id.as_str()).collect();
    let mut attributes: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    let mut evidence = BTreeSet::new();
    for f in &slice.facts {
        if !entities.contains(f.subject.as_str()) {
            continue;
        }
        let status = f.current().map_or(FactStatus::Active, |v| v.status);
        match f.object_entity.as_deref().filter(|o| entities.contains(o)) {
            Some(object) => g.edges.push(Edge {
                id: f.id.clone(),
                source: f.subject.clone(),
                target: object.to_owned(),
                kind: "fact",
                label: f.predicate.replace('_', " "),
            }),
            None => attributes
                .entry(&f.subject)
                .or_default()
                .push(serde_json::json!({
                    "predicate": f.predicate.replace('_', " "),
                    "object": f.object,
                    "status": status.as_str(),
                })),
        }
        for e in &f.evidence {
            if sources.contains(e.source_id.as_str())
                && evidence.insert((f.subject.as_str(), e.source_id.as_str()))
            {
                g.edges.push(Edge {
                    id: format!("{}>{}", f.subject, e.source_id),
                    source: f.subject.clone(),
                    target: e.source_id.clone(),
                    kind: "evidence",
                    label: String::new(),
                });
            }
        }
    }
    for e in &slice.entities {
        let mut data = BTreeMap::new();
        if !e.aliases.is_empty() {
            data.insert("aliases", serde_json::json!(e.aliases));
        }
        if let Some(attrs) = attributes.remove(e.id.as_str()) {
            data.insert("attributes", attrs.into());
        }
        g.nodes.push(Node {
            id: e.id.clone(),
            kind: "entity",
            label: e.name.clone(),
            group: e.kind.clone(),
            data,
        });
    }
    add_memory(&mut g, slice, memory);
    g
}

fn add_memory(g: &mut Graph, slice: &GraphSlice, memory: MemoryTree<'_>) {
    let live: Vec<&FileView> = memory.files.iter().filter(|f| f.file.is_live()).collect();
    if live.is_empty() {
        return;
    }
    let used: BTreeSet<Option<&str>> = live
        .iter()
        .map(|f| f.file.meta.category_id.as_deref())
        .collect();
    for c in memory.categories {
        if used.contains(&c.id.as_deref()) {
            g.nodes.push(Node {
                id: category_id(c.id.as_deref()),
                kind: "category",
                label: c.display_name.clone(),
                group: c.display_name.clone(),
                data: BTreeMap::new(),
            });
        }
    }
    let names: Vec<(String, &str)> = slice
        .entities
        .iter()
        .flat_map(|e| {
            std::iter::once(&e.name)
                .chain(&e.aliases)
                .filter(|n| n.chars().count() >= MIN_MENTION_CHARS)
                .map(move |n| (n.to_lowercase(), e.id.as_str()))
        })
        .collect();
    let facts: BTreeMap<&str, &keep_knowledge::model::Fact> =
        slice.facts.iter().map(|f| (f.id.as_str(), f)).collect();
    for f in live {
        let id = f.file.memory_id.clone();
        let meta = &f.file.meta;
        let label = if meta.display_name.is_empty() {
            meta.path.clone()
        } else {
            meta.display_name.clone()
        };
        let mut data = BTreeMap::new();
        data.insert("path", meta.path.as_str().into());
        if !meta.description.is_empty() {
            data.insert("description", meta.description.as_str().into());
        }
        data.insert("entries", f.revision.entries.len().into());
        let category = category_id(meta.category_id.as_deref());
        g.nodes.push(Node {
            id: id.clone(),
            kind: "file",
            label,
            group: category.clone(),
            data,
        });
        g.edges.push(Edge {
            id: format!("{id}>{category}"),
            source: id.clone(),
            target: category,
            kind: "in_category",
            label: String::new(),
        });
        let mut linked = BTreeSet::new();
        let cited = f
            .revision
            .entries
            .iter()
            .flat_map(|e| &e.annotation.sources)
            .filter_map(|s| s.strip_prefix("knowledge://fact/"))
            .filter_map(|fact| facts.get(fact))
            .flat_map(|fact| {
                std::iter::once(fact.subject.as_str()).chain(fact.object_entity.as_deref())
            });
        for entity in cited {
            if linked.insert(entity) {
                g.edges.push(Edge {
                    id: format!("{id}>{entity}"),
                    source: id.clone(),
                    target: entity.to_owned(),
                    kind: "cites",
                    label: String::new(),
                });
            }
        }
        let text = f.revision.text().to_lowercase();
        for (name, entity) in &names {
            if !linked.contains(entity) && mentions(&text, name) {
                linked.insert(entity);
                g.edges.push(Edge {
                    id: format!("{id}>{entity}"),
                    source: id.clone(),
                    target: (*entity).to_owned(),
                    kind: "mentions",
                    label: String::new(),
                });
            }
        }
    }
}

/// Whether `name` occurs in `text` as whole words.
fn mentions(text: &str, name: &str) -> bool {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
    text.match_indices(name).any(|(at, m)| {
        boundary(text[..at].chars().next_back()) && boundary(text[at + m.len()..].chars().next())
    })
}

fn collection_id(name: &str) -> String {
    format!("collection:{name}")
}

fn category_id(id: Option<&str>) -> String {
    format!("category:{}", id.unwrap_or("other"))
}

#[cfg(test)]
mod tests {
    use super::mentions;

    #[test]
    fn mentions_match_whole_words_only() {
        assert!(mentions("ask hermione granger.", "hermione granger"));
        assert!(mentions("ron", "ron"));
        assert!(!mentions("ronald", "ron"));
        assert!(!mentions("the baron's", "ron"));
    }
}
