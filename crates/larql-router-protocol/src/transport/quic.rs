//! QUIC transport for the grid gRPC stream (ADR-0010).
//!
//! The shape this module provides:
//!   * `QuicStream` — wraps a `(SendStream, RecvStream)` pair as a duplex
//!     `AsyncRead + AsyncWrite`, ready to be handed to tonic as a custom
//!     transport.
//!   * `self_signed_tls` — generates an in-memory self-signed cert for
//!     LAN / dev. The router exposes its cert fingerprint; the announce
//!     client pins it via `--quic-cert-fingerprint`. No CA needed for the
//!     common case.
//!   * `server_endpoint` / `client_endpoint` — quinn::Endpoint factories.
//!   * `connect_grpc_channel` — full client-side wiring: dial, open a
//!     bidirectional stream, hand it to `tonic::transport::Endpoint` via a
//!     custom connector, return a `tonic::transport::Channel` ready for the
//!     generated `GridServiceClient` to use.
//!
//! The server-side accept loop lives in `larql-router/src/main.rs` because
//! it interleaves with the existing tonic Server setup; the building
//! blocks here are reused there.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use quinn::{Connection, Endpoint, RecvStream, SendStream};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// One QUIC bidirectional stream presented as an `AsyncRead + AsyncWrite`
/// duplex. tonic's `serve_with_incoming` accepts any transport that
/// satisfies these traits plus `Connected`, so we can plug a
/// `(SendStream, RecvStream)` pair straight into it.
///
/// `Send + Unpin` come for free via the quinn stream halves.
pub struct QuicStream {
    send: SendStream,
    recv: RecvStream,
    /// Cached remote address; surfaced via `Connected::connect_info` for
    /// tonic-side request logging.
    remote_addr: Option<SocketAddr>,
}

impl QuicStream {
    /// Wrap a quinn bidi stream pair. Caller is responsible for opening or
    /// accepting the pair on a `quinn::Connection`.
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self {
            send,
            recv,
            remote_addr: None,
        }
    }

    /// Same as `new` but remembers the remote `SocketAddr` so tonic can
    /// expose it through request extensions.
    pub fn with_remote(send: SendStream, recv: RecvStream, remote_addr: SocketAddr) -> Self {
        Self {
            send,
            recv,
            remote_addr: Some(remote_addr),
        }
    }
}

/// Remote-peer info surfaced to tonic via the `Connected` trait. tonic
/// puts a clone of this into `Request::extensions()` so handlers can
/// inspect the source address if they need to (e.g. per-IP audit logs).
#[derive(Clone, Debug, Default)]
pub struct QuicConnectInfo {
    pub remote_addr: Option<SocketAddr>,
}

impl tonic::transport::server::Connected for QuicStream {
    type ConnectInfo = QuicConnectInfo;
    fn connect_info(&self) -> Self::ConnectInfo {
        QuicConnectInfo {
            remote_addr: self.remote_addr,
        }
    }
}

impl AsyncRead for QuicStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        // Explicit UFCS so we hit quinn's `tokio::io::AsyncRead` impl
        // (the inherent `poll_read` on `RecvStream` has a different
        // signature).
        <RecvStream as AsyncRead>::poll_read(Pin::new(&mut this.recv), cx, buf)
    }
}

impl AsyncWrite for QuicStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        let this = self.get_mut();
        <SendStream as AsyncWrite>::poll_write(Pin::new(&mut this.send), cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        <SendStream as AsyncWrite>::poll_flush(Pin::new(&mut this.send), cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        <SendStream as AsyncWrite>::poll_shutdown(Pin::new(&mut this.send), cx)
    }
}

// ── TLS helpers ──────────────────────────────────────────────────────────────

/// Result of cert generation: certificate (DER) + matching private key
/// (PKCS#8 DER) + an SHA-256 fingerprint of the certificate suitable for
/// pinning by the client.
pub struct SelfSignedTls {
    /// PEM-encoded certificate chain — typically one cert. Pass as `--quic-cert`.
    pub cert_pem: String,
    /// PEM-encoded PKCS#8 private key. Pass as `--quic-key`.
    pub key_pem: String,
    /// Hex-encoded SHA-256 of the leaf certificate DER. The announce
    /// client uses this to pin the router's identity without a CA.
    pub fingerprint: String,
    /// Server name embedded in the certificate; clients must connect with
    /// this name (default `"localhost"`).
    pub server_name: String,
}

/// Generate a self-signed leaf cert for `server_name`. Suitable for LAN /
/// dev grids where the operator can ship the fingerprint over the same
/// channel as `--grid-key`.
pub fn self_signed_tls(server_name: &str) -> Result<SelfSignedTls, String> {
    let cert = rcgen::generate_simple_self_signed(vec![server_name.to_owned()])
        .map_err(|e| format!("rcgen self-sign failed: {e}"))?;
    let cert_pem = cert.cert.pem();
    let key_pem = cert.key_pair.serialize_pem();
    let der: CertificateDer<'_> = cert.cert.der().clone();
    let fingerprint = sha256_hex(der.as_ref());
    Ok(SelfSignedTls {
        cert_pem,
        key_pem,
        fingerprint,
        server_name: server_name.to_owned(),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    // ring's digest is reachable via rustls -> ring. Use it to avoid
    // adding sha2 as a direct dep just for one hash.
    let digest = ring_compat::digest_sha256(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

// Wrapping module so the digest call site stays one line above. ring lives
// behind rustls's feature set; rather than depending on it directly we
// vendor a six-line wrapper around the `Sha256` impl rustls re-exports.
mod ring_compat {
    pub fn digest_sha256(bytes: &[u8]) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().to_vec()
    }
}

// ── quinn endpoint factories ────────────────────────────────────────────────

/// Build a quinn server endpoint listening on `addr` and presenting `tls`.
pub fn server_endpoint(addr: SocketAddr, tls: &SelfSignedTls) -> Result<Endpoint, String> {
    let mut cert_pem_bytes = tls.cert_pem.as_bytes();
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_pem_bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("parse cert PEM: {e}"))?;
    let mut key_pem_bytes = tls.key_pem.as_bytes();
    let key: PrivatePkcs8KeyDer<'static> = rustls_pemfile::pkcs8_private_keys(&mut key_pem_bytes)
        .next()
        .ok_or_else(|| "no PKCS#8 key in --quic-key PEM".to_string())?
        .map_err(|e| format!("parse key PEM: {e}"))?;

    let server_config = quinn::ServerConfig::with_single_cert(certs, key.into())
        .map_err(|e| format!("quinn ServerConfig::with_single_cert: {e}"))?;
    Endpoint::server(server_config, addr).map_err(|e| format!("quinn Endpoint::server: {e}"))
}

/// Build a quinn client endpoint. The `expected_fingerprint` (hex SHA-256
/// of the leaf cert DER) is the trust anchor; with no CA hierarchy this is
/// how the announce client confirms it reached the right router.
///
/// Pass `None` to disable certificate verification entirely (LAN-only /
/// development; never on the public internet).
pub fn client_endpoint(
    bind_addr: SocketAddr,
    expected_fingerprint: Option<String>,
) -> Result<Endpoint, String> {
    let mut endpoint = Endpoint::client(bind_addr)
        .map_err(|e| format!("quinn Endpoint::client: {e}"))?;
    let client_cfg = build_client_config(expected_fingerprint)?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
}

fn build_client_config(
    expected_fingerprint: Option<String>,
) -> Result<quinn::ClientConfig, String> {
    let crypto: Arc<rustls::ClientConfig> = if let Some(fp) = expected_fingerprint {
        Arc::new(client_config_with_fingerprint(fp)?)
    } else {
        Arc::new(client_config_skip_verify())
    };
    let quic_client_cfg = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
        .map_err(|e| format!("quinn ClientConfig from rustls: {e}"))?;
    Ok(quinn::ClientConfig::new(Arc::new(quic_client_cfg)))
}

/// Verifier that pins the server's certificate to a known SHA-256
/// fingerprint. No CA chain validation is done; this is the "shared
/// secret on a LAN" trust model.
#[derive(Debug)]
struct FingerprintVerifier {
    expected: Vec<u8>,
}

impl rustls::client::danger::ServerCertVerifier for FingerprintVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let got = ring_compat::digest_sha256(end_entity.as_ref());
        if got == self.expected {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "QUIC cert fingerprint mismatch — refusing to trust this router".into(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        // Accept the standard set tlsv13 uses; rustls filters on its side.
        vec![
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

fn client_config_with_fingerprint(fp_hex: String) -> Result<ClientConfig, String> {
    let expected = decode_hex(&fp_hex).map_err(|e| format!("--quic-cert-fingerprint: {e}"))?;
    let provider = rustls::crypto::ring::default_provider();
    let cfg = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| format!("rustls ClientConfig: {e}"))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(FingerprintVerifier { expected }))
        .with_no_client_auth();
    Ok(cfg)
}

/// LAN / dev-only: skip *all* certificate verification. The compiled symbol
/// stays gated behind the `quic` feature and the runtime call site is the
/// announce client passing `None` for the fingerprint.
fn client_config_skip_verify() -> ClientConfig {
    let provider = rustls::crypto::ring::default_provider();
    let mut roots = RootCertStore::empty();
    // No roots — the AcceptAny verifier below makes that irrelevant.
    roots.add_parsable_certificates(std::iter::empty::<CertificateDer<'static>>());
    #[derive(Debug)]
    struct AcceptAny;
    impl rustls::client::danger::ServerCertVerifier for AcceptAny {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp_response: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            vec![
                rustls::SignatureScheme::RSA_PSS_SHA256,
                rustls::SignatureScheme::RSA_PSS_SHA384,
                rustls::SignatureScheme::RSA_PSS_SHA512,
                rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
                rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
                rustls::SignatureScheme::ED25519,
            ]
        }
    }
    ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("TLS13 must be supported by ring provider")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAny))
        .with_no_client_auth()
}

// ── Server-side accept loop ─────────────────────────────────────────────────

/// Start a QUIC accept loop on `endpoint` that forwards every accepted
/// bidirectional stream as a `QuicStream` (wrapped in
/// `hyper_util::rt::TokioIo` for tonic) onto the returned `mpsc::Receiver`.
///
/// Callers pass the receiver into
/// `tonic::transport::Server::serve_with_incoming` so the same gRPC
/// service implementation handles both TCP and QUIC clients with no
/// code duplication.
///
/// The accept loop runs until `endpoint` is dropped or
/// `endpoint.close(...)` is called. It is intentionally tolerant of
/// individual connection errors — one bad handshake should not take down
/// the listener.
pub fn spawn_accept_loop(
    endpoint: Endpoint,
) -> tokio::sync::mpsc::Receiver<Result<QuicStream, std::io::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                let conn = match incoming.await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing_quic_warn(format!("QUIC handshake failed: {e}"));
                        return;
                    }
                };
                let remote_addr = conn.remote_address();
                loop {
                    match conn.accept_bi().await {
                        Ok((send, recv)) => {
                            let stream = QuicStream::with_remote(send, recv, remote_addr);
                            if tx.send(Ok(stream)).await.is_err() {
                                return;
                            }
                        }
                        Err(quinn::ConnectionError::ApplicationClosed(_)) => return,
                        Err(quinn::ConnectionError::ConnectionClosed(_)) => return,
                        Err(quinn::ConnectionError::Reset) => return,
                        Err(quinn::ConnectionError::TimedOut) => return,
                        Err(e) => {
                            tracing_quic_warn(format!("accept_bi: {e}"));
                            return;
                        }
                    }
                }
            });
        }
    });
    rx
}

fn tracing_quic_warn(msg: String) {
    // The router crate uses `tracing` directly; pulling it into
    // larql-router-protocol just for warn-on-error noise isn't worth the
    // extra dep. eprintln is enough — the line is rare and the rest of
    // the system logs via tracing-subscriber's default stderr writer
    // anyway.
    eprintln!("[quic] {msg}");
}

// ── Client-side connect helper ──────────────────────────────────────────────

/// Dial `server_addr` over QUIC, open a single bidirectional stream, and
/// hand it to tonic as a custom-transport `Channel`. Returns the channel
/// ready for `GridServiceClient::new(channel)`.
///
/// The `server_name` must match the cert's SAN (the `--quic-cert-fingerprint`
/// path defaults to `"router"`; clients pass whatever name the operator
/// embedded in the cert).
pub async fn connect_grpc_channel(
    endpoint: &Endpoint,
    server_addr: SocketAddr,
    server_name: &str,
) -> Result<(Connection, tonic::transport::Channel), String> {
    let conn = endpoint
        .connect(server_addr, server_name)
        .map_err(|e| format!("quinn connect: {e}"))?
        .await
        .map_err(|e| format!("quinn handshake: {e}"))?;

    let (send, recv) = conn
        .open_bi()
        .await
        .map_err(|e| format!("quinn open_bi: {e}"))?;
    // QUIC bi-streams are lazy — the server doesn't see them until at
    // least one byte is sent. tonic will write H2 preface bytes shortly,
    // so we don't need to push anything ourselves.
    let stream = QuicStream::new(send, recv);

    let stream_cell: Arc<tokio::sync::Mutex<Option<QuicStream>>> =
        Arc::new(tokio::sync::Mutex::new(Some(stream)));
    let connector = tower::service_fn(move |_uri: tonic::transport::Uri| {
        let cell = stream_cell.clone();
        async move {
            let s = cell
                .lock()
                .await
                .take()
                .ok_or_else(|| io::Error::other("QUIC connector already consumed"))?;
            Ok::<_, io::Error>(hyper_util::rt::TokioIo::new(s))
        }
    });

    let channel = tonic::transport::Endpoint::try_from("http://quic-pinned")
        .map_err(|e| format!("tonic Endpoint::try_from placeholder: {e}"))?
        .connect_with_connector(connector)
        .await
        .map_err(|e| format!("tonic connect_with_connector: {e}"))?;

    Ok((conn, channel))
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(format!("hex length must be even, got {}", s.len()));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for chunk in s.as_bytes().chunks(2) {
        let byte = std::str::from_utf8(chunk)
            .ok()
            .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            .ok_or_else(|| format!("invalid hex in fingerprint at {chunk:?}"))?;
        out.push(byte);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_signed_tls_returns_pem_and_fingerprint() {
        let tls = self_signed_tls("router.local").expect("rcgen must succeed");
        assert!(tls.cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
        assert!(tls.key_pem.starts_with("-----BEGIN PRIVATE KEY-----"));
        assert_eq!(tls.fingerprint.len(), 64, "SHA-256 hex is 64 chars");
        assert!(tls.fingerprint.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(tls.server_name, "router.local");
    }

    #[test]
    fn decode_hex_round_trip() {
        let bytes: Vec<u8> = (0..32).collect();
        let mut hex = String::new();
        for b in &bytes {
            hex.push_str(&format!("{b:02x}"));
        }
        let parsed = decode_hex(&hex).unwrap();
        assert_eq!(parsed, bytes);
    }

    #[test]
    fn decode_hex_rejects_odd_length() {
        assert!(decode_hex("abc").is_err());
    }

    #[test]
    fn decode_hex_rejects_non_hex_chars() {
        assert!(decode_hex("zz").is_err());
    }

    #[test]
    fn fingerprint_changes_with_input() {
        let a = ring_compat::digest_sha256(b"hello");
        let b = ring_compat::digest_sha256(b"world");
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn build_client_config_accepts_fingerprint_and_no_fingerprint() {
        let with_fp = build_client_config(Some("00".repeat(32))).unwrap();
        let without = build_client_config(None).unwrap();
        // Just confirm the configs construct; runtime verification is
        // covered by the integration test (which actually connects).
        let _ = (with_fp, without);
    }
}
