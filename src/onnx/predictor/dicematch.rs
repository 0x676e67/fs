use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;
pub struct DicematchMatchPredictor(ImageClassifierPredictor);

impl DicematchMatchPredictor {
    /// Create a new instance of the DicematchMatchPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("dicematch.onnx", config).await?,
        ))
    }
}

impl Predictor for DicematchMatchPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
