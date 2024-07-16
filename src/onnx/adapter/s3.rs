use std::path::PathBuf;
use std::sync::Arc;

use super::FetchAdapter;
use crate::error::Error;
use crate::onnx::adapter::progress;
use crate::Result;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_s3::config::{Credentials, Region, SharedCredentialsProvider};
use aws_sdk_s3::Client;
use tokio::fs;
use tokio::sync::OnceCell;

static S3_CONFIG: OnceCell<SdkConfig> = OnceCell::const_new();

/// A struct providing most necessary APIs to work with Cloudflare S3 object storage.
#[derive(Debug, Clone)]
pub struct S3Adapter {
    bucket_name: String,
    prefix_key: Option<String>,
    client: Arc<Client>,
}

impl S3Adapter {
    /// Creates a new instance of R2Manager. The region is set to us-east-1 which aliases
    /// to auto. Read more here <https://developers.cloudflare.com/r2/api/s3/api/>.
    pub async fn new(
        bucket_name: String,
        prefix_key: Option<String>,
        url: String,
        client_id: String,
        secret: String,
    ) -> S3Adapter {
        // Load AWS SDK configuration
        let s3_config = S3_CONFIG
            .get_or_init(|| async {
                let creds = Credentials::new(client_id, secret, None, None, "onnx");
                aws_config::load_defaults(BehaviorVersion::v2024_03_28())
                    .await
                    .into_builder()
                    .endpoint_url(url)
                    .credentials_provider(SharedCredentialsProvider::new(creds))
                    .region(Region::new("auto"))
                    .build()
            })
            .await;

        S3Adapter {
            bucket_name,
            prefix_key,
            client: Arc::new(aws_sdk_s3::Client::new(s3_config)),
        }
    }

    async fn sha256(&self, filepath: &PathBuf) -> Option<String> {
        if filepath.exists() {
            tracing::info!("{} exists, skipping download", filepath.display());
            fs::read_to_string(&filepath).await.ok()
        } else {
            let filename = filepath.file_name()?.to_str()?;
            self.download_file(filename, filepath).await.ok()?;
            fs::read_to_string(filepath).await.ok()
        }
    }

    async fn download_file(&self, key: &str, model_file: &PathBuf) -> Result<()> {
        // Prefix key with the prefix_key if it exists
        let key = self
            .prefix_key
            .as_ref()
            .map(|prefix_key| format!("{}/{}", prefix_key, key))
            .unwrap_or_else(|| key.to_string());

        // Get object from the bucket
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(key)
            .send()
            .await
            .map_err(|e| Error::CloudflareR2SdkError(e.to_string()))?;

        // Get content length
        let len = resp.content_length().unwrap_or(0) as u64;

        // IntoAsyncRead is implemented for `impl AsyncRead + Unpin + Send + Sync`
        let stream = resp.body.into_async_read();

        // Create a progress bar
        let pb = progress::ProgressBar::new(len)?;

        // Open file for writing
        let mut tmp_file = fs::File::create(model_file).await?;

        // Copy the stream to the file
        tokio::io::copy(&mut pb.wrap_async_read(stream), &mut tmp_file).await?;
        drop(tmp_file);

        Ok(())
    }
}

impl FetchAdapter for S3Adapter {
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: std::path::PathBuf,
        update_check: bool,
    ) -> Result<PathBuf> {
        // Create model directory if it does not exist
        if !model_dir.exists() {
            tracing::info!("creating model directory: {}", model_dir.display());
            fs::create_dir_all(&model_dir).await?;
        }

        // Build version sha256 filepath
        let sha256_filename = model_dir.join(format!("{model_name}.sha256"));

        // check version.json is exist
        if sha256_filename.exists() && update_check {
            tracing::info!("deleting {}", sha256_filename.display());
            fs::remove_file(&sha256_filename).await?;
        }

        // Build model file path
        let model_file = model_dir.join(model_name);

        // If model file does not exist or update_check is true, download the model
        if !model_file.exists() {
            // Download model
            self.download_file(model_name, &model_file).await?;
        }

        // If update_check is true, check the hash of the model
        if update_check {
            if let Some(expected_hash) = self.sha256(&sha256_filename).await {
                let current_hash = Self::file_sha256(&model_file).await?;
                if current_hash.ne(&expected_hash) {
                    tracing::info!(
                        "model {} hash mismatch, downloading...",
                        model_file.display()
                    );
                    self.download_file(model_name, &model_file).await?;
                }
            }
        }

        Ok(model_file)
    }
}
