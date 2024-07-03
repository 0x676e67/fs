use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct RockstackPredictor(ImagePairClassifierPredictor);

impl RockstackPredictor {
    /// Create a new instance of the RockstackPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("rockstack.onnx", None, config, false).await?,
        ))
    }
}

impl Predictor for RockstackPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
