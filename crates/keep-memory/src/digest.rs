//! Keyed digests for idempotency receipts.
//!
//! Receipts store an HMAC-SHA256 of the canonical request instead of the
//! request, so a receipt never holds memory text and cannot be used to
//! confirm a guess about forgotten text without the server key.

use sha2::{Digest, Sha256};

const BLOCK: usize = 64;

pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(message);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(inner);
    hex::encode(outer.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_rfc_4231_case_2() {
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
