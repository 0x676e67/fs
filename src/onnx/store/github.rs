use crate::{constant, error::Error, Result};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokio::fs;

use super::{file_sha256, Fetch};

pub struct GithubStore;

impl Fetch for GithubStore {
    async fn fetch_model(
        &self,
        model_name: &'static str,
        model_dir: PathBuf,
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

        // If model file does not exist or update_check is true, download the model
        if !model_file.exists() || update_check {
            // Download model
            download_file(&model_url, &model_file).await?;
        }

        // If update_check is true, check the hash of the model
        if update_check {
            if let Some(version_info) = version_info(&version_info_path).await {
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
                    download_file(&model_url, &model_file).await?;
                }
            }
        }

        Ok(model_file)
    }
}

async fn version_info(version_info_path: &PathBuf) -> Option<HashMap<String, String>> {
    // Check if version.json exists
    let version_info: HashMap<String, String> = if version_info_path.exists() {
        let data = fs::read_to_string(&version_info_path).await.ok()?;
        serde_json::from_str(&data).ok()?
    } else {
        // Download version.json
        download_file(constant::GITHUB_VERSION_INFO_URL, constant::VERSION_INFO)
            .await
            .ok()?;
        let data = fs::read_to_string(version_info_path).await.ok()?;
        serde_json::from_str(&data).ok()?
    };

    Some(version_info)
}

async fn download_file<P: AsRef<Path>>(url: &str, filename: P) -> Result<()> {
    let bytes = reqwest::get(url).await?.bytes().await?;
    let mut out = fs::File::create(filename.as_ref()).await?;
    let bytes_count = tokio::io::copy(&mut bytes.as_ref(), &mut out).await?;
    drop(out);
    drop(bytes);
    tracing::info!(
        "downloaded {} bytes to {}",
        bytes_count,
        filename.as_ref().display()
    );
    Ok(())
}
