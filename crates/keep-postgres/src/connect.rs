//! Opening Postgres connections, locally or to a remote managed database.
//!
//! Every Keep crate that talks to Postgres connects through [`connect`], so
//! one TLS policy applies everywhere:
//!
//! - `sslmode=disable`: plaintext. Use it only on a private network, such as
//!   a Compose network.
//! - `sslmode=prefer`: TLS if the server offers it, else plaintext. Honoured
//!   only for loopback hosts and Unix sockets; for any other host it is
//!   treated as `require`, so a remote connection is never downgraded.
//! - `sslmode=require`, `verify-ca`, `verify-full`: TLS is required.
//!
//! Without `sslmode`, loopback hosts and Unix sockets use `prefer` and every
//! other host `require`. Whenever TLS is used the server certificate and host
//! name are verified, unlike libpq's `require`. Trusted roots are the
//! Mozilla set plus, if given, the PEM file in `sslrootcert`.

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;
use tokio_postgres::config::{Host, SslMode};
use tokio_postgres::{Client, Config, NoTls};
use tokio_postgres_rustls::MakeRustlsConnect;

#[derive(Debug)]
pub struct ConnectError(String);

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConnectError {}

impl From<tokio_postgres::Error> for ConnectError {
    fn from(e: tokio_postgres::Error) -> Self {
        Self(e.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tls {
    Disable,
    Prefer,
    Require,
}

/// A connection URL split into what tokio-postgres understands and Keep's
/// TLS settings.
#[derive(Debug)]
pub struct Target {
    pub config: Config,
    pub tls: Tls,
    pub root_cert: Option<PathBuf>,
}

/// Parse `url` and decide the TLS policy. Accepts `postgres://` URLs and
/// libpq key=value strings.
pub fn target(url: &str) -> Result<Target, ConnectError> {
    let (rest, mode, root_cert) = take_tls_params(url)?;
    let config =
        Config::from_str(&rest).map_err(|e| ConnectError(format!("invalid Postgres URL: {e}")))?;
    let local = is_local(&config);
    let tls = match mode.as_deref() {
        Some("disable") => Tls::Disable,
        Some("prefer") | Some("allow") | None if local => Tls::Prefer,
        Some("prefer") | Some("allow") | None => Tls::Require,
        Some("require") | Some("verify-ca") | Some("verify-full") => Tls::Require,
        Some(other) => {
            return Err(ConnectError(format!(
                "unsupported sslmode {other:?}: use disable, prefer, require, verify-ca or verify-full"
            )));
        }
    };
    Ok(Target {
        config,
        tls,
        root_cert,
    })
}

/// Connect with Keep's TLS policy. The connection runs on its own task until
/// the client is dropped; connection errors surface on the next query.
pub async fn connect(url: &str) -> Result<Client, ConnectError> {
    let Target {
        mut config,
        tls,
        root_cert,
    } = target(url)?;
    if tls == Tls::Disable {
        config.ssl_mode(SslMode::Disable);
        let (client, conn) = config.connect(NoTls).await?;
        tokio::spawn(async move {
            let _ = conn.await;
        });
        return Ok(client);
    }
    config.ssl_mode(match tls {
        Tls::Prefer => SslMode::Prefer,
        _ => SslMode::Require,
    });
    let connector = MakeRustlsConnect::new(client_config(root_cert.as_deref())?);
    let (client, conn) = config.connect(connector).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    Ok(client)
}

fn client_config(root_cert: Option<&Path>) -> Result<rustls::ClientConfig, ConnectError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    if let Some(path) = root_cert {
        let certs = CertificateDer::pem_file_iter(path)
            .and_then(|it| it.collect::<Result<Vec<_>, _>>())
            .map_err(|e| ConnectError(format!("reading sslrootcert {}: {e}", path.display())))?;
        if certs.is_empty() {
            return Err(ConnectError(format!(
                "sslrootcert {} holds no certificates",
                path.display()
            )));
        }
        for c in certs {
            roots
                .add(c)
                .map_err(|e| ConnectError(format!("sslrootcert {}: {e}", path.display())))?;
        }
    }
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| ConnectError(e.to_string()))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(config)
}

fn is_local(config: &Config) -> bool {
    let hosts = config.get_hosts();
    !hosts.is_empty()
        && hosts.iter().all(|h| match h {
            Host::Tcp(name) => {
                name.eq_ignore_ascii_case("localhost")
                    || name
                        .trim_matches(['[', ']'])
                        .parse::<IpAddr>()
                        .is_ok_and(|ip| ip.is_loopback())
            }
            #[cfg(unix)]
            Host::Unix(_) => true,
        })
}

/// Remove `sslmode` and `sslrootcert` from `url`, returning them separately,
/// because tokio-postgres rejects `verify-*` modes and `sslrootcert`.
fn take_tls_params(url: &str) -> Result<(String, Option<String>, Option<PathBuf>), ConnectError> {
    let mut mode = None;
    let mut root = None;
    if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        let Some((base, query)) = url.split_once('?') else {
            return Ok((url.to_owned(), None, None));
        };
        let mut kept = Vec::new();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            match k {
                "sslmode" => mode = Some(v.to_ascii_lowercase()),
                "sslrootcert" => root = Some(PathBuf::from(percent_decode(v)?)),
                _ => kept.push(pair),
            }
        }
        let rest = if kept.is_empty() {
            base.to_owned()
        } else {
            format!("{base}?{}", kept.join("&"))
        };
        return Ok((rest, mode, root));
    }
    let mut kept = Vec::new();
    for part in url.split_whitespace() {
        match part.split_once('=') {
            Some(("sslmode", v)) => mode = Some(v.trim_matches('\'').to_ascii_lowercase()),
            Some(("sslrootcert", v)) => root = Some(PathBuf::from(v.trim_matches('\''))),
            _ => kept.push(part),
        }
    }
    Ok((kept.join(" "), mode, root))
}

fn percent_decode(s: &str) -> Result<String, ConnectError> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .ok_or_else(|| ConnectError("bad percent-encoding in sslrootcert".into()))?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| ConnectError("sslrootcert is not UTF-8".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tls(url: &str) -> Tls {
        target(url).unwrap().tls
    }

    #[test]
    fn remote_hosts_never_fall_back_to_plaintext() {
        assert_eq!(tls("postgres://u:p@db.example.com/keep"), Tls::Require);
        assert_eq!(
            tls("postgres://u:p@db.example.com/keep?sslmode=prefer"),
            Tls::Require
        );
        assert_eq!(tls("host=10.0.0.5 user=u dbname=keep"), Tls::Require);
        assert_eq!(
            tls("postgres://u:p@db.example.com/keep?sslmode=verify-full"),
            Tls::Require
        );
    }

    #[test]
    fn loopback_prefers_and_disable_is_explicit() {
        assert_eq!(tls("postgres://u:p@127.0.0.1:15433/keep"), Tls::Prefer);
        assert_eq!(tls("postgres://u:p@localhost/keep"), Tls::Prefer);
        assert_eq!(tls("postgres://u:p@[::1]:5432/keep"), Tls::Prefer);
        assert_eq!(
            tls("postgres://u:p@postgres:5432/keep?sslmode=disable"),
            Tls::Disable
        );
        assert_eq!(
            tls("postgres://u:p@127.0.0.1/keep?sslmode=require"),
            Tls::Require
        );
    }

    #[test]
    fn tls_parameters_are_removed_and_others_kept() {
        let t = target(
            "postgres://u:p@db.example.com/keep?sslmode=verify-full&sslrootcert=%2Fetc%2Fca.pem&options=-csearch_path%3Dx",
        )
        .unwrap();
        assert_eq!(t.root_cert.as_deref(), Some(Path::new("/etc/ca.pem")));
        assert_eq!(t.config.get_options(), Some("-csearch_path=x"));
        assert!(target("postgres://u@h/keep?sslmode=bogus").is_err());
    }
}
