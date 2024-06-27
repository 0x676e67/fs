use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct HandNumberPuzzlePredictor(ImageClassifierPredictor);

impl HandNumberPuzzlePredictor {
    /// Create a new instance of the HandNumberPuzzlePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("hand_number_puzzle.onnx", None, config).await?,
        ))
    }
}

impl Predictor for HandNumberPuzzlePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
