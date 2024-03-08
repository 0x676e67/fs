use anyhow::Result;
use image::DynamicImage;
use ndarray::Array4;
use ort::{GraphOptimizationLevel, Session};
use sha2::Digest;
use sha2::Sha256;
use std::sync::OnceLock;
use std::{
    collections::HashMap,
    f32, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use crate::homedir;
use crate::BootArgs;

use super::image_processing::check_input_image_size;
use super::image_processing::process_classifier_image;
use super::image_processing::process_pair_classifier_ans_image;
use super::image_processing::process_pair_classifier_image;

static INIT_VERSION: OnceLock<()> = OnceLock::new();

pub struct ImagePairClassifierPredictor((Session, bool));
pub struct ImageClassifierPredictor(Session);

impl ImagePairClassifierPredictor {
    /// Create a new instance of the ImagePairClassifierPredictor
    pub fn new(onnx: &'static str, args: &BootArgs, is_grayscale: bool) -> Result<Self> {
        Ok(Self((create_model_session(onnx, args)?, is_grayscale)))
    }
}

impl ImageClassifierPredictor {
    /// Create a new instance of the ImageClassifierPredictor
    pub fn new(onnx: &'static str, args: &BootArgs) -> Result<Self> {
        Ok(Self(create_model_session(onnx, args)?))
    }
}

impl ImagePairClassifierPredictor {
    /// Run prediction on the model
    pub fn run_prediction(&self, left: Array4<f32>, right: Array4<f32>) -> Result<Vec<f32>> {
        let inputs = ort::inputs! {
            "input_left" => left,
            "input_right" => right,
        }?;

        let outputs = self.0 .0.run(inputs)?;
        let output = outputs[0]
            .extract_tensor::<f32>()?
            .view()
            .t()
            .into_owned()
            .into_iter()
            .collect();
        return Ok(output);
    }

    #[inline]
    pub fn predict(&self, mut image: DynamicImage) -> Result<i32> {
        check_input_image_size(&image)?;

        let mut max_prediction = f32::NEG_INFINITY;
        let width = image.width();
        let mut max_index = 0;
        let left = process_pair_classifier_ans_image(&mut image, (52, 52), self.0 .1)?;

        for i in 0..(width / 200) {
            let right = process_pair_classifier_image(&image, (0, i), (52, 52), self.0 .1)?;
            let prediction = self.run_prediction(left.clone(), right)?;
            let prediction_value = prediction[0];
            if prediction_value > max_prediction {
                max_prediction = prediction_value;
                max_index = i;
            }
        }
        Ok(max_index as i32)
    }
}

impl ImageClassifierPredictor {
    fn run_prediction(&self, image: Array4<f32>) -> Result<Vec<f32>> {
        let outputs = self.0.run(ort::inputs! {
            "input" => image,
        }?)?;
        let output = outputs[0]
            .extract_tensor::<f32>()?
            .view()
            .t()
            .into_owned()
            .into_iter()
            .collect();
        Ok(output)
    }

    #[inline]
    pub fn predict(&self, mut image: DynamicImage) -> Result<i32> {
        let mut max_prediction = f32::NEG_INFINITY;
        let mut max_index: i32 = -1;

        for i in 0..6 {
            let ts = process_classifier_image(&mut image, i, (52, 52))?;

            let prediction = self.run_prediction(ts)?;
            let prediction_value = prediction[0];
            if prediction_value > max_prediction {
                max_prediction = prediction_value;
                max_index = i as i32;
            }
        }

        Ok(max_index)
    }
}

fn create_model_session(onnx: &'static str, args: &BootArgs) -> Result<Session> {
    let model_dir = args
        .model_dir
        .as_ref()
        .map(|x| x.to_owned())
        .unwrap_or_else(|| {
            homedir::home_dir()
                .unwrap_or(PathBuf::new())
                .join(".funcaptcha_models")
        });

    if INIT_VERSION.get().is_none() {
        // check version.json is exist
        if model_dir.join("version.json").exists() {
            // delete version.json
            fs::remove_file(model_dir.join("version.json"))?;
        }
        let _ = INIT_VERSION.set(());
    }

    let model_file = initialize_model(onnx, model_dir, args.update_check)?;
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(true)?
        .with_memory_pattern(true)?
        .with_intra_threads(args.num_threads as i16)?
        .with_allocator(args.allocator)?
        .with_model_from_file(model_file)?;
    Ok(session)
}

fn initialize_model(
    model_name: &'static str,
    model_dir: PathBuf,
    update_check: bool,
) -> Result<String> {
    let model_filename = format!("{}/{model_name}", model_dir.display());

    // Create parent directory if it does not exist
    if let Some(parent_dir) = Path::new(&model_filename).parent() {
        if !parent_dir.exists() {
            fs::create_dir_all(&parent_dir)?;
        }
    }

    let version_url = "https://github.com/MagicalMadoka/funcaptcha-challenger/releases/download/model/version.json";
    let model_url = format!(
        "https://github.com/MagicalMadoka/funcaptcha-challenger/releases/download/model/{model_name}",
    );

    let version_json_path = format!("{}/version.json", model_dir.display());

    // Check if version.json exists
    let version_info = if PathBuf::from(&version_json_path).exists() && !update_check {
        if update_check {
            tracing::info!("checking for model update...");
            download_file(&version_url, &version_json_path)?;
        }
        let info: HashMap<String, String> =
            serde_json::from_str(&fs::read_to_string(&version_json_path)?)?;
        info
    } else {
        tracing::info!("version.json not found, downloading...");
        download_file(&version_url, &version_json_path)?;
        let info: HashMap<String, String> =
            serde_json::from_str(&fs::read_to_string(version_json_path)?)?;
        info
    };

    if !Path::new(&model_filename).exists() || update_check {
        tracing::info!("model {model_name} not found, downloading...");
        download_file(&model_url, &model_filename)?;

        let expected_hash = &version_info[&model_name
            .split(".")
            .next()
            .ok_or_else(|| anyhow::anyhow!("model name is not valid"))?
            .to_string()];

        tracing::info!("expected hash: {}", expected_hash);
        let current_hash = file_sha256(&model_filename)?;
        tracing::info!("current hash: {}", current_hash);

        if expected_hash.ne(&current_hash) {
            tracing::info!("model {} hash mismatch, downloading...", model_filename);
            download_file(&model_url, &model_filename)?;
        }
    }

    Ok(model_filename)
}

fn download_file(url: &str, filename: &str) -> Result<()> {
    let mut response = reqwest::blocking::get(url)?;
    let mut out = fs::File::create(filename)?;
    let mut buffer = [0; 1024];
    tracing::info!("downloading {}...", filename);
    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        out.write_all(&buffer[..n])?;
    }
    drop(out);
    tracing::info!("downloaded {} done", filename);
    Ok(())
}

fn file_sha256(filename: &str) -> Result<String> {
    let mut file = std::fs::File::open(filename)?;
    let mut sha256 = Sha256::new();
    let mut buffer = [0; 1024];
    while let Ok(n) = file.read(&mut buffer) {
        if n == 0 {
            break;
        }
        sha256.update(&buffer[..n]);
    }
    Ok(format!("{:x}", sha256.finalize()))
}
