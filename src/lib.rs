pub mod alloc;
#[cfg(target_family = "unix")]
pub mod daemon;
pub mod error;
pub mod homedir;
pub mod onnx;
pub mod serve;
pub mod update;

use clap::{Args, Parser, Subcommand};
use error::Error;
pub use homedir::setting_dir;
use std::{net::SocketAddr, path::PathBuf};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Parser)]
#[clap(author, version, about, arg_required_else_help = true)]
#[command(args_conflicts_with_subcommands = true)]
pub struct Opt {
    #[clap(subcommand)]
    pub commands: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run server
    Run(BootArgs),
    /// Start server daemon
    #[cfg(target_family = "unix")]
    Start(BootArgs),
    /// Restart server daemon
    #[cfg(target_family = "unix")]
    Restart(BootArgs),
    /// Stop server daemon
    #[cfg(target_family = "unix")]
    Stop,
    /// Show the server daemon log
    #[cfg(target_family = "unix")]
    Log,
    /// Show the server daemon process
    #[cfg(target_family = "unix")]
    PS,
    /// Update the application
    Update,
}

#[derive(Args, Clone, Debug)]
pub struct BootArgs {
    /// Debug mode
    #[clap(short, long)]
    pub debug: bool,

    /// Bind address
    #[clap(short, long, default_value = "0.0.0.0:8000")]
    pub bind: SocketAddr,

    /// TLS certificate file
    #[clap(long)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key file
    #[clap(long)]
    pub tls_key: Option<PathBuf>,

    /// Export API key
    #[clap(short = 'A', long)]
    pub api_key: Option<String>,

    /// Multiple image submission limits
    #[clap(short = 'L', long, default_value = "3")]
    pub limit: usize,

    /// Funcaptcha model update check
    #[clap(short = 'U', long)]
    pub update_check: bool,

    /// Funcaptcha model directory
    #[clap(short = 'M', long)]
    pub model_dir: Option<PathBuf>,

    /// Number of threads (ONNX Runtime)
    #[clap(short = 'N', long, default_value = "1")]
    pub num_threads: u16,

    /// Execution provider allocator e.g. device, arena (ONNX Runtime)
    #[clap(long, default_value = "device", value_parser = alloc_parser)]
    pub allocator: ort::AllocatorType,

    /// Fallback solver, supported: "yescaptcha / capsolver"
    #[clap(short = 'S', long)]
    pub fallback_solver: Option<String>,

    /// Fallback solver client key
    #[clap(short = 'K', long, requires = "fallback_solver")]
    pub fallback_key: Option<String>,

    /// Fallback solver endpoint
    #[clap(short = 'E', long, requires = "fallback_solver")]
    pub fallback_endpoint: Option<String>,

    /// Fallback solver image limit
    #[clap(short = 'D', long, requires = "fallback_solver", default_value = "1")]
    pub fallback_image_limit: usize,

    #[clap(subcommand)]
    pub store: onnx::Config,
}

fn alloc_parser(s: &str) -> Result<ort::AllocatorType> {
    match s {
        "device" => Ok(ort::AllocatorType::Device),
        "arena" => Ok(ort::AllocatorType::Arena),
        _ => Err(Error::InvalidAllocator(s.to_string())),
    }
}
