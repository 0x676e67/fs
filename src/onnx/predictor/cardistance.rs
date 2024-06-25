use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct CardistancePredictor(ImagePairClassifierPredictor);

impl CardistancePredictor {
    /// Create a new instance of the CardistancePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("cardistance.onnx", config, false).await?,
        ))
    }
}

impl Predictor for CardistancePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
