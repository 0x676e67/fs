use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct LumberLengthGamePredictor(ImageClassifierPredictor);

impl LumberLengthGamePredictor {
    /// Create a new instance of the LumberLengthGamePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("lumber-length-game.onnx", config).await?,
        ))
    }
}

impl Predictor for LumberLengthGamePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
