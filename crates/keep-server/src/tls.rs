//! HTTPS for `keep serve`: rustls with the ring provider, from PEM files.

use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustls::ServerConfig;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;

/// How long a client gets to finish the handshake.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// A server configuration from a PEM certificate chain and private key.
pub fn server_config(cert: &Path, key: &Path) -> anyhow::Result<Arc<ServerConfig>> {
    let certs = CertificateDer::pem_file_iter(cert)
        .and_then(|it| it.collect::<Result<Vec<_>, _>>())
        .map_err(|e| anyhow::anyhow!("reading TLS certificate {}: {e}", cert.display()))?;
    anyhow::ensure!(
        !certs.is_empty(),
        "TLS certificate {} holds no certificates",
        cert.display()
    );
    let key = PrivateKeyDer::from_pem_file(key)
        .map_err(|e| anyhow::anyhow!("reading TLS key {}: {e}", key.display()))?;
    let mut config =
        ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| anyhow::anyhow!("TLS certificate and key do not fit: {e}"))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

/// Accepts TCP connections and completes TLS handshakes in their own tasks,
/// so one slow client cannot hold up the others.
pub struct TlsListener {
    ready: mpsc::Receiver<(TlsStream<TcpStream>, SocketAddr)>,
    local: SocketAddr,
    acceptor: JoinHandle<()>,
}

impl TlsListener {
    pub fn new(tcp: TcpListener, config: Arc<ServerConfig>) -> io::Result<Self> {
        let local = tcp.local_addr()?;
        let tls = TlsAcceptor::from(config);
        let (tx, ready) = mpsc::channel(64);
        let acceptor = tokio::spawn(async move {
            loop {
                let (stream, addr) = match tcp.accept().await {
                    Ok(c) => c,
                    Err(_) => {
                        // Usually out of file descriptors; back off briefly.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let (tls, tx) = (tls.clone(), tx.clone());
                tokio::spawn(async move {
                    // A failed or slow handshake just drops the connection.
                    if let Ok(Ok(s)) =
                        tokio::time::timeout(HANDSHAKE_TIMEOUT, tls.accept(stream)).await
                    {
                        let _ = tx.send((s, addr)).await;
                    }
                });
            }
        });
        Ok(Self {
            ready,
            local,
            acceptor,
        })
    }
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        self.acceptor.abort();
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.ready.recv().await {
            Some(c) => c,
            // The acceptor task never drops its sender while running.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local)
    }
}
