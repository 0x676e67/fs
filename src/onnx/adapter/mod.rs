pub mod github;
mod progress;
pub mod r2;

use std::path::PathBuf;

use crate::Result;
use clap::Subcommand;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;

/// Trait defining the fetch operation for ONNX models.
pub trait FetchAdapter {
    /// Asynchronously fetches an ONNX model.
    ///
    /// # Parameters
    /// - `model_name`: The name of the model to fetch.
    /// - `model_dir`: The directory where the model should be stored.
    /// - `update_check`: A boolean indicating whether to check for updates to the model.
    ///
    /// # Returns
    /// - A `Result` containing a `String`. On success, this `String` is the path to the fetched model.
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: std::path::PathBuf,
        update_check: bool,
    ) -> Result<PathBuf>;
}

/// Enum representing the ONNX model storage options.
/// Currently, there are two options: R2 and Github.
#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum ONNXFetchConfig {
    /// Represents the CloudFlare R2 storage option.
    R2 {
        /// The name of the bucket.
        #[clap(short = 'b', long)]
        bucket_name: String,

        /// The bucket key prefix.
        #[clap(short = 'p', long)]
        prefix_key: Option<String>,

        /// The URL of the Cloudflare KV.
        #[clap(short = 'l', long)]
        url: String,

        /// The client ID of the Cloudflare KV.
        #[clap(short = 'c', long)]
        client_id: String,

        /// The secret of the Cloudflare KV.
        #[clap(short = 's', long)]
        secret: String,
    },
    /// Represents the Github storage option.
    Github,
}

impl Default for ONNXFetchConfig {
    fn default() -> Self {
        ONNXFetchConfig::Github
    }
}

pub enum ONNXFetch {
    R2(r2::R2Adapter),
    Github(github::GithubAdapter),
}

impl ONNXFetch {
    pub async fn new(onnx_store: ONNXFetchConfig) -> Self {
        match onnx_store {
            ONNXFetchConfig::R2 {
                bucket_name,
                prefix_key,
                url: cloudflare_kv_uri,
                client_id: cloudflare_kv_client_id,
                secret: cloudflare_kv_secret,
            } => {
                let r2 = r2::R2Adapter::new(
                    bucket_name,
                    prefix_key,
                    cloudflare_kv_uri,
                    cloudflare_kv_client_id,
                    cloudflare_kv_secret,
                )
                .await;
                ONNXFetch::R2(r2)
            }
            ONNXFetchConfig::Github => ONNXFetch::Github(github::GithubAdapter),
        }
    }
}

impl Default for ONNXFetch {
    fn default() -> Self {
        ONNXFetch::Github(github::GithubAdapter)
    }
}

impl FetchAdapter for ONNXFetch {
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: std::path::PathBuf,
        update_check: bool,
    ) -> Result<PathBuf> {
        match self {
            ONNXFetch::R2(r2) => r2.fetch_model(model_name, model_dir, update_check).await,
            ONNXFetch::Github(github) => {
                github
                    .fetch_model(model_name, model_dir, update_check)
                    .await
            }
        }
    }
}

async fn file_sha256(filename: &PathBuf) -> Result<String> {
    let mut file = tokio::fs::File::open(filename).await?;
    let mut sha256 = Sha256::new();
    let mut buffer = [0; 1024];
    while let Ok(n) = file.read(&mut buffer).await {
        if n == 0 {
            break;
        }
        sha256.update(&buffer[..n]);
    }
    Ok(format!("{:x}", sha256.finalize()))
}
