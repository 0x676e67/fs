use std::path::PathBuf;
use std::sync::Arc;

use super::Fetch;
use crate::{constant, Result};
use aws_config::SdkConfig;
use aws_sdk_s3::config::Region;
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
        cloudflare_kv_uri: String,
        cloudflare_kv_client_id: String,
        cloudflare_kv_secret: String,
    ) -> R2Store {
        std::env::set_var("R2_ACCESS_KEY_ID", cloudflare_kv_client_id);
        std::env::set_var("R2_SECRET_ACCESS_KEY", cloudflare_kv_secret);
        let s3_config = S3_CONFIG
            .get_or_init(|| async {
                aws_config::load_from_env()
                    .await
                    .into_builder()
                    .endpoint_url(cloudflare_kv_uri)
                    .region(Region::new("us-east-1"))
                    .build()
            })
            .await;

        R2Store {
            bucket_name: bucket_name.into(),
            client: Arc::new(aws_sdk_s3::Client::new(s3_config)),
        }
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

        // Build model url
        let model_url =
            format!("https://github.com/0x676e67/fs/releases/download/model/{model_name}");

        // check version.json is exist
        if version_info_path.exists() && update_check {
            tracing::info!("deleting {}", version_info_path.display());
            fs::remove_file(&version_info_path).await?;
        }

        // Build model file path
        let model_file = model_dir.join(model_name);

        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(model_name)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Failed to get object: {}", e);
                crate::Error::CloudflareR2SdkError(e.to_string())
            })?;

        // tokio::fs::write(&model_filename, resp.body).await?;

        Ok(PathBuf::new())
    }
}