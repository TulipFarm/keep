//! Keep's server: one [`KeepService`] over the Memory and Knowledge
//! engines, served to coding agents over MCP (stdio) and a JSON HTTP API.

pub mod api;
pub mod auth;
pub mod backend;
pub mod budget;
pub mod config;
pub mod embedder;
pub mod error;
pub mod http;
pub mod mcp;
pub mod scope;
pub mod service;
pub mod settings;
pub mod tls;

pub use backend::{Backend, HttpBackend, ScopeDefaults};
pub use error::{ApiError, ErrorCode};
pub use scope::{Scope, ScopeArg};
pub use service::{KeepService, ServiceOptions};
