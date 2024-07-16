use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct Maze2Predictor(ImageClassifierPredictor);

impl Maze2Predictor {
    /// Create a new instance of the Maze2Predictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("maze2.onnx", None, config).await?,
        ))
    }
}

impl Predictor for Maze2Predictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
