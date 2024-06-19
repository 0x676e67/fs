use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct LumberLengthGamePredictor(ImagePairClassifierPredictor);

impl LumberLengthGamePredictor {
    /// Create a new instance of the LumberLengthGamePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("lumber-length-game.onnx", config, false).await?,
        ))
    }
}

impl Predictor for LumberLengthGamePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
