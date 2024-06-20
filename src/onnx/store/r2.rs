use std::sync::Arc;

use aws_config::SdkConfig;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::Client;
use tokio::sync::OnceCell;

use super::Fetch;

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

    /// Get the bucket name of the R2Manager.
    pub fn get_bucket_name(&self) -> &str {
        &self.bucket_name
    }

    /// Get an object in Vec<u8> form.
    pub async fn get(&self, object_name: &str) -> Option<Vec<u8>> {
        let get_request = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(object_name)
            .send()
            .await;

        if get_request.is_ok() {
            let result = get_request.unwrap();
            tracing::debug!("{:?}", result);
            tracing::info!("Got successfully {} from {}", object_name, self.bucket_name);
            let bytes = result.body.collect().await.unwrap().into_bytes().to_vec();
            return Some(bytes);
        } else {
            tracing::debug!("{:?}", get_request.unwrap_err());
            tracing::error!("Unable to get {} from {}.", object_name, self.bucket_name);
            None
        }
    }
}

impl Fetch for R2Store {
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: std::path::PathBuf,
        update_check: bool,
    ) -> crate::Result<String> {
        // Create model directory if it does not exist
        if !model_dir.exists() {
            tracing::info!("creating model directory: {}", model_dir.display());
            tokio::fs::create_dir_all(&model_dir).await?;
        }

        let model_filename = format!("{}/{}", model_dir.display(), model_name);

        // Create parent directory if it does not exist
        if let Some(parent_dir) = std::path::Path::new(&model_filename).parent() {
            if !parent_dir.exists() {
                tokio::fs::create_dir_all(parent_dir).await?;
            }
        }

        let model = self.get(model_name).await.unwrap();

        tokio::fs::write(&model_filename, &model).await?;

        Ok(model_filename)
    }
}
