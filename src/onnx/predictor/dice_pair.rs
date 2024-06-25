use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;
pub struct DicePairPredictor(ImageClassifierPredictor);

impl DicePairPredictor {
    /// Create a new instance of the DicePairPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("dice_pair.onnx", config).await?,
        ))
    }
}

impl Predictor for DicePairPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
