use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct ShadowsPredictor(ImageClassifierPredictor);

impl ShadowsPredictor {
    /// Create a new instance of the TrainCoordinatesPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("shadows.onnx", config).await?,
        ))
    }
}

impl Predictor for ShadowsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
