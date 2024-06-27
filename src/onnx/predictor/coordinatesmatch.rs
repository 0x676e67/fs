use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;
pub struct CoordinatesMatchPredictor(ImagePairClassifierPredictor);

impl CoordinatesMatchPredictor {
    /// Create a new instance of the CoordinatesMatchPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("coordinatesmatch.onnx", None, config, false).await?,
        ))
    }
}

impl Predictor for CoordinatesMatchPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
