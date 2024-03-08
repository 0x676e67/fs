pub mod alloc;
#[cfg(target_family = "unix")]
pub mod daemon;
pub mod homedir;
pub mod model;
pub mod serve;
pub mod update;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
pub use homedir::setting_dir;
use std::{net::SocketAddr, path::PathBuf};

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
    /// Show the server daemon process
    #[cfg(target_family = "unix")]
    Status,
    /// Show the server daemon log
    #[cfg(target_family = "unix")]
    Log,
    /// Update the application
    Update,
}

#[derive(Args, Clone)]
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

    /// API key
    #[clap(short = 'A', long)]
    pub api_key: Option<String>,

    /// Multiple image submission limits
    #[clap(short = 'M', long, default_value = "3")]
    pub multi_image_limit: usize,

    /// Funcaptcha model update check
    #[clap(short = 'U', long)]
    pub update_check: bool,

    /// Funcaptcha model directory
    #[clap(long)]
    pub model_dir: Option<PathBuf>,

    /// Number of threads (ONNX Runtime)
    #[clap(long, default_value = "1")]
    pub num_threads: u16,

    /// Execution provider allocator e.g. device, arena (ONNX Runtime)
    #[clap(long, default_value = "device", value_parser = alloc_parser)]
    pub allocator: ort::AllocatorType,

    /// Fallback solver, supported: "yescaptcha / capsolver"
    #[clap(short = 'S', long, default_value = "yescaptcha")]
    pub fallback_solver: Option<String>,

    /// Fallback solver client key
    #[clap(short = 'K', long, requires = "fallback_solver")]
    pub fallback_key: Option<String>,

    /// Fallback solver endpoint
    #[clap(short = 'E', long, requires = "fallback_solver")]
    pub fallback_endpoint: Option<String>,

    /// Fallback image limit
    #[clap(short = 'D', long, requires = "fallback_solver", default_value = "1")]
    pub fallback_image_limit: usize,
}

fn alloc_parser(s: &str) -> anyhow::Result<ort::AllocatorType> {
    match s {
        "device" => Ok(ort::AllocatorType::Device),
        "arena" => Ok(ort::AllocatorType::Arena),
        _ => Err(anyhow::anyhow!("Invalid allocator type")),
    }
}
