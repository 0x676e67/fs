pub mod github;
mod progress;
pub mod s3;

use std::path::{Path, PathBuf};

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

    /// Asynchronously fetches the SHA256 hash of a file.
    /// # Parameters
    /// - `filepath`: The path to the file.
    /// # Returns
    /// - A `Result` containing a `String`. On success, this `String` is the SHA256 hash of the file.
    async fn file_sha256(filepath: impl AsRef<Path>) -> Result<String> {
        let mut file = tokio::fs::File::open(filepath).await?;
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
}

/// Enum representing the ONNX model storage options.
/// Currently, there are two options: S3 and Github.
#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum Config {
    /// Represents the AWS S3 storage option.
    S3 {
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

impl Default for Config {
    fn default() -> Self {
        Config::Github
    }
}

pub enum Adapter {
    S3(s3::S3Adapter),
    Github(github::GithubAdapter),
}

impl Adapter {
    pub async fn new(config: Config) -> Self {
        match config {
            Config::S3 {
                bucket_name,
                prefix_key,
                url: cloudflare_kv_uri,
                client_id: cloudflare_kv_client_id,
                secret: cloudflare_kv_secret,
            } => {
                let r2 = s3::S3Adapter::new(
                    bucket_name,
                    prefix_key,
                    cloudflare_kv_uri,
                    cloudflare_kv_client_id,
                    cloudflare_kv_secret,
                )
                .await;
                Adapter::S3(r2)
            }
            Config::Github => Adapter::Github(github::GithubAdapter),
        }
    }
}

impl Default for Adapter {
    fn default() -> Self {
        Adapter::Github(github::GithubAdapter)
    }
}

impl FetchAdapter for Adapter {
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: std::path::PathBuf,
        update_check: bool,
    ) -> Result<PathBuf> {
        match self {
            Adapter::S3(r2) => r2.fetch_model(model_name, model_dir, update_check).await,
            Adapter::Github(github) => {
                github
                    .fetch_model(model_name, model_dir, update_check)
                    .await
            }
        }
    }
}
