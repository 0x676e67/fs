use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct PenguinsPredictor(ImageClassifierPredictor);

impl PenguinsPredictor {
    /// Create a new instance of the PenguinsPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("penguin.onnx", None, config).await?,
        ))
    }
}

impl Predictor for PenguinsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
