use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::{file_sha256, Fetch};
use crate::error::Error;
use crate::{constant, Result};
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, SdkConfig};
use aws_sdk_s3::config::{Credentials, Region, SharedCredentialsProvider};
use aws_sdk_s3::Client;
use tokio::fs;
use tokio::sync::OnceCell;

static S3_CONFIG: OnceCell<SdkConfig> = OnceCell::const_new();

/// A struct providing most necessary APIs to work with Cloudflare R2 object storage.
#[derive(Debug, Clone)]
pub struct R2Store {
    bucket_name: String,
    client: Arc<Client>,
}

impl R2Store {
    /// Creates a new instance of R2Manager. The region is set to us-east-1 which aliases
    /// to auto. Read more here <https://developers.cloudflare.com/r2/api/s3/api/>.
    pub async fn new(
        bucket_name: String,
        url: String,
        client_id: String,
        secret: String,
    ) -> R2Store {
        // Load AWS SDK configuration
        let s3_config = S3_CONFIG
            .get_or_init(|| async {
                let creds = Credentials::new(client_id, secret, None, None, "arkose-onnx");
                aws_config::load_defaults(BehaviorVersion::v2024_03_28())
                    .await
                    .into_builder()
                    .endpoint_url(url)
                    .credentials_provider(SharedCredentialsProvider::new(creds))
                    .timeout_config(
                        TimeoutConfig::builder()
                            .connect_timeout(std::time::Duration::from_secs(10))
                            .read_timeout(std::time::Duration::from_secs(10))
                            .build(),
                    )
                    .region(Region::new("auto"))
                    .build()
            })
            .await;

        R2Store {
            bucket_name: bucket_name.into(),
            client: Arc::new(aws_sdk_s3::Client::new(s3_config)),
        }
    }

    async fn version_info(&self, version_info_path: &PathBuf) -> Option<HashMap<String, String>> {
        // Check if version.json exists
        let version_info: HashMap<String, String> = if version_info_path.exists() {
            let data = fs::read_to_string(&version_info_path).await.ok()?;
            serde_json::from_str(&data).ok()?
        } else {
            // Download version.json
            self.download_file(constant::VERSION_INFO, version_info_path)
                .await
                .ok()?;
            let data = fs::read_to_string(version_info_path).await.ok()?;
            serde_json::from_str(&data).ok()?
        };

        Some(version_info)
    }

    async fn download_file(&self, key: &str, model_file: &PathBuf) -> Result<()> {
        // Get object from the bucket
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(key)
            .send()
            .await
            .map_err(|e| Error::CloudflareR2SdkError(e.to_string()))?;

        // IntoAsyncRead is implemented for `impl AsyncRead + Unpin + Send + Sync`
        let mut stream = resp.body.into_async_read();

        // Open file for writing
        let mut out = fs::File::create(model_file).await?;

        // Copy the stream to the file
        let copy = tokio::io::copy(&mut stream, &mut out).await?;
        drop(out);
        drop(stream);
        tracing::info!("downloaded {} bytes to {}", copy, model_file.display());

        Ok(())
    }
}

impl Fetch for R2Store {
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

        // Build version info path
        let version_info_path = model_dir.join(constant::VERSION_INFO);

        // check version.json is exist
        if version_info_path.exists() && update_check {
            tracing::info!("deleting {}", version_info_path.display());
            fs::remove_file(&version_info_path).await?;
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
            if let Some(version_info) = self.version_info(&version_info_path).await {
                // Get model name without extension
                let model_name = model_name
                    .split('.')
                    .next()
                    .ok_or_else(|| Error::InvalidModelName(model_name.to_string()))?;

                // Get expected hash from version.json
                let expected_hash = version_info
                    .get(model_name)
                    .ok_or_else(|| Error::InvalidModelVersionInfo(model_name.to_string()))?;

                let current_hash = file_sha256(&model_file).await?;

                if current_hash.ne(expected_hash) {
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
