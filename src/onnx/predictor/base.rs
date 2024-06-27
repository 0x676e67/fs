use crate::{
    constant,
    error::Error,
    homedir,
    onnx::{
        adapter::FetchAdapter,
        util::{
            check_input_image_size, process_classifier_image, process_pair_classifier_ans_image,
            process_pair_classifier_image,
        },
        ONNXConfig,
    },
    Result,
};
use image::DynamicImage;
use ndarray::Array4;
#[cfg(feature = "cuda")]
use ort::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::DirectMLExecutionProvider;
#[cfg(feature = "rocm")]
use ort::ROCmExecutionProvider;
use ort::{GraphOptimizationLevel, MemoryInfo, Session};
use std::f32;
use tokio::sync::OnceCell;

pub struct ImageClassifierPredictor {
    input_shape: (u32, u32),
    session: OnceCell<Session>,
    active: OnceCell<()>,
}

impl ImageClassifierPredictor {
    pub async fn new(
        onnx: &'static str,
        input_shape: Option<(u32, u32)>,
        config: &ONNXConfig,
    ) -> Result<Self> {
        let predictor = ImageClassifierPredictor {
            session: OnceCell::new(),
            active: OnceCell::new(),
            input_shape: input_shape.unwrap_or((52, 52)),
        };

        // If the session is created successfully, set the session and wait for it to be initialized
        match create_onnx_session(onnx, config).await {
            Ok(session) => {
                let _ = predictor.session.set(session);
                let _ = predictor.active.set(());
            }
            Err(err) => {
                tracing::warn!("Failed to create session: {}", err);
            }
        }

        Ok(predictor)
    }

    #[inline]
    pub fn active(&self) -> bool {
        self.active.get().is_some()
    }

    fn run_prediction(&self, image: Array4<f32>) -> Result<Vec<f32>> {
        let outputs = self
            .session
            .get()
            .ok_or_else(|| Error::OnnxSessionNotInitialized)?
            .run(ort::inputs! {
                "input" => image,
            }?)?;
        let output = outputs[0]
            .try_extract_tensor::<f32>()?
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
            let ts = process_classifier_image(&mut image, i, self.input_shape)?;

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

pub struct ImagePairClassifierPredictor {
    input_shape: (u32, u32),
    session: OnceCell<Session>,
    active: OnceCell<()>,
    is_grayscale: bool,
}

impl ImagePairClassifierPredictor {
    pub async fn new(
        onnx: &'static str,
        input_shape: Option<(u32, u32)>,
        config: &ONNXConfig,
        is_grayscale: bool,
    ) -> Result<Self> {
        let predictor = ImagePairClassifierPredictor {
            session: OnceCell::new(),
            active: OnceCell::new(),
            input_shape: input_shape.unwrap_or((52, 52)),
            is_grayscale,
        };

        // If the session is created successfully, set the session and wait for it to be initialized
        match create_onnx_session(onnx, config).await {
            Ok(session) => {
                let _ = predictor.session.set(session);
                let _ = predictor.active.set(());
            }
            Err(err) => {
                tracing::warn!("Failed to create session: {}", err);
            }
        }

        Ok(predictor)
    }

    #[inline]
    pub fn active(&self) -> bool {
        self.active.get().is_some()
    }

    /// Run prediction on the model
    pub fn run_prediction(&self, left: Array4<f32>, right: Array4<f32>) -> Result<Vec<f32>> {
        let inputs = ort::inputs! {
            "input_left" => left,
            "input_right" => right,
        }?;

        let outputs = self
            .session
            .get()
            .ok_or_else(|| Error::OnnxSessionNotInitialized)?
            .run(inputs)?;
        let output = outputs[0]
            .try_extract_tensor::<f32>()?
            .into_owned()
            .into_iter()
            .collect();
        Ok(output)
    }

    #[inline]
    pub fn predict(&self, mut image: DynamicImage) -> Result<i32> {
        check_input_image_size(&image)?;

        let mut max_prediction = f32::NEG_INFINITY;
        let width = image.width();
        let mut max_index = 0;
        let left =
            process_pair_classifier_ans_image(&mut image, self.input_shape, self.is_grayscale)?;

        for i in 0..(width / 200) {
            let right =
                process_pair_classifier_image(&image, (0, i), self.input_shape, self.is_grayscale)?;
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

async fn create_onnx_session(onnx: &'static str, config: &ONNXConfig) -> Result<Session> {
    let model_dir = config
        .model_dir
        .as_ref()
        .map(|x| x.to_owned())
        .unwrap_or_else(|| {
            homedir::home_dir()
                .unwrap_or_default()
                .join(constant::MODEL_DIR)
        });

    // Fetch the model file
    let model_file = config
        .onnx_store
        .fetch_model(onnx, model_dir, config.update_check)
        .await?;

    // Create a new session
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(true)?
        .with_memory_pattern(true)?
        .with_intra_threads(config.num_threads as usize)?
        .with_execution_providers([
            // Prefer TensorRT over CUDA.
            #[cfg(feature = "cuda")]
            CUDAExecutionProvider::default().build(),
            // Use DirectML on Windows if NVIDIA EPs are not available
            #[cfg(feature = "directml")]
            DirectMLExecutionProvider::default().build(),
            // Or use ANE on Apple platforms
            #[cfg(feature = "coreml")]
            CoreMLExecutionProvider::default().build(),
            // Or use rocm on AMD platforms
            #[cfg(feature = "rocm")]
            ROCmExecutionProvider::default().build(),
        ])?
        .with_allocator(MemoryInfo::new_cpu(
            config.allocator,
            ort::MemoryType::Default,
        )?)?
        .commit_from_file(model_file)?;
    Ok(session)
}
