use super::{base::ImageClassifierPredictor, Predictor};
use crate::onnx::ONNXConfig;
use crate::Result;
use image::DynamicImage;
pub struct FrankenheadPredictor(ImageClassifierPredictor);

impl FrankenheadPredictor {
    /// Create a new instance of the Frankenhead
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("frankenhead.onnx", config).await?,
        ))
    }
}

impl Predictor for FrankenheadPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
