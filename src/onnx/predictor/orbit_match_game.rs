use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct OrbitMatchGamePredictor(ImagePairClassifierPredictor);

impl OrbitMatchGamePredictor {
    /// Create a new instance of the OrbitMatchGamePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("orbit_match_game.onnx", None, config, false).await?,
        ))
    }
}

impl Predictor for OrbitMatchGamePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
