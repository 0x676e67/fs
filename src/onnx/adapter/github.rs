use crate::{onnx::adapter::progress, Result};
use std::path::{Path, PathBuf};
use tokio::fs;

use super::FetchAdapter;

pub struct GithubAdapter(pub String);

impl Default for GithubAdapter {
    fn default() -> Self {
        Self("https://github.com/0x676e67/fs/releases/download/model".to_owned())
    }
}

impl FetchAdapter for GithubAdapter {
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

        // Build version sha256 filepath
        let sha256_filename = model_dir.join(format!("{model_name}.sha256"));

        // Build model url
        let model_url = format!("{}/{model_name}", self.0);

        // check version.json is exist
        if sha256_filename.exists() && update_check {
            tracing::info!("deleting {}", sha256_filename.display());
            fs::remove_file(&sha256_filename).await?;
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
            if let Some(expected_hash) = sha256(&self.0, &sha256_filename).await {
                let current_hash = Self::file_sha256(&model_file).await?;
                if current_hash.ne(&expected_hash) {
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

async fn download_file(url: &str, filepath: impl AsRef<Path>) -> Result<()> {
    use futures_util::StreamExt;

    // Fetch the response
    let resp = reqwest::get(url).await?;

    // Get the response content length
    let content_length = resp.content_length().unwrap_or(0);

    // Read the response bytes stream
    let mut byte_stream = resp.bytes_stream();

    // Create a progress bar
    let pb = progress::ProgressBar::new(content_length)?;

    // Create a file
    let mut tmp_file = fs::File::create(filepath.as_ref()).await?;

    while let Some(item) = byte_stream.next().await {
        let bytes = item?;
        let mut read = pb.wrap_async_read(bytes.as_ref());
        tokio::io::copy(&mut read, &mut tmp_file).await?;
    }

    drop(tmp_file);
    Ok(())
}

async fn sha256(url: &str, filepath: &PathBuf) -> Option<String> {
    if filepath.exists() {
        tracing::info!("{} exists, skipping download", filepath.display());
        fs::read_to_string(&filepath).await.ok()
    } else {
        let filename = filepath.file_name()?.to_str()?;
        let url = format!("{url}/{filename}");
        download_file(&url, filepath).await.ok()?;
        fs::read_to_string(&filepath).await.ok()
    }
}
