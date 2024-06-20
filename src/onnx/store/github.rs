use crate::Result;
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
    ) -> Result<String> {
        // Create model directory if it does not exist
        if !model_dir.exists() {
            tracing::info!("creating model directory: {}", model_dir.display());
            fs::create_dir_all(&model_dir).await?;
        }

        let model_filename = format!("{}/{model_name}", model_dir.display());

        // Create parent directory if it does not exist
        if let Some(parent_dir) = Path::new(&model_filename).parent() {
            if !parent_dir.exists() {
                fs::create_dir_all(parent_dir).await?;
            }
        }

        let version_url = "https://github.com/0x676e67/fs/releases/download/model/version.json";
        let model_url =
            format!("https://github.com/0x676e67/fs/releases/download/model/{model_name}",);

        let version_json_path = format!("{}/version.json", model_dir.display());

        // Check if version.json exists
        let version_info = if PathBuf::from(&version_json_path).exists() {
            let info: HashMap<String, String> =
                serde_json::from_str(&fs::read_to_string(&version_json_path).await?)?;
            info
        } else {
            download_file(version_url, &version_json_path).await?;
            let info: HashMap<String, String> =
                serde_json::from_str(&fs::read_to_string(version_json_path).await?)?;
            info
        };

        if !Path::new(&model_filename).exists() || update_check {
            download_file(&model_url, &model_filename).await?;

            let expected_hash = &version_info[&model_name
                .split('.')
                .next()
                .ok_or_else(|| crate::Error::InvalidModelName(model_name.to_string()))?
                .to_string()];

            let current_hash = file_sha256(&model_filename).await?;

            if expected_hash.ne(&current_hash) {
                tracing::info!("model {} hash mismatch, downloading...", model_filename);
                download_file(&model_url, &model_filename).await?;
            }
        }

        Ok(model_filename)
    }
}

async fn download_file(url: &str, filename: &str) -> Result<()> {
    let bytes = reqwest::get(url).await?.bytes().await?;
    let mut out = fs::File::create(filename).await?;
    tokio::io::copy(&mut bytes.as_ref(), &mut out).await?;
    drop(out);
    drop(bytes);
    tracing::info!("downloaded {} done", filename);
    Ok(())
}
