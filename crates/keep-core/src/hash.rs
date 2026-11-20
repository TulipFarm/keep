//! Content hashing used to pin datasets, manifests and artifacts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// A SHA-256 digest rendered as `sha256:<64 lowercase hex chars>`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContentHash(String);

impl ContentHash {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(format!("sha256:{}", sha256_hex(bytes)))
    }

    /// The bare hex digest without the `sha256:` prefix.
    pub fn hex(&self) -> &str {
        &self.0["sha256:".len()..]
    }

    pub fn short(&self) -> &str {
        &self.hex()[..12]
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid content hash {0:?}: expected sha256:<64 lowercase hex>")]
pub struct InvalidHash(String);

impl TryFrom<String> for ContentHash {
    type Error = InvalidHash;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let hex_part = value
            .strip_prefix("sha256:")
            .ok_or_else(|| InvalidHash(value.clone()))?;
        let valid = hex_part.len() == 64
            && hex_part
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if valid {
            Ok(Self(value))
        } else {
            Err(InvalidHash(value))
        }
    }
}

impl std::str::FromStr for ContentHash {
    type Err = InvalidHash;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(s.to_owned())
    }
}

impl From<ContentHash> for String {
    fn from(value: ContentHash) -> Self {
        value.0
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Serialise a JSON value with object keys sorted at every depth and no
/// insignificant whitespace, so equal values always hash equally.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).expect("string serialises"));
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_ignores_key_order() {
        let a: serde_json::Value =
            serde_json::from_str(r#"{"b":1,"a":{"y":[1,2],"x":null}}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{"a":{"x":null,"y":[1,2]},"b":1}"#).unwrap();
        assert_eq!(canonical_json(&a), canonical_json(&b));
        assert_eq!(canonical_json(&a), r#"{"a":{"x":null,"y":[1,2]},"b":1}"#);
    }

    #[test]
    fn content_hash_rejects_malformed_values() {
        assert!("sha256:abc".parse::<ContentHash>().is_err());
        assert!(
            format!("md5:{}", "0".repeat(64))
                .parse::<ContentHash>()
                .is_err()
        );
        assert!(
            format!("sha256:{}", "A".repeat(64))
                .parse::<ContentHash>()
                .is_err()
        );
        let ok = ContentHash::of_bytes(b"keep");
        assert_eq!(ok.as_str().parse::<ContentHash>().unwrap(), ok);
    }
}
