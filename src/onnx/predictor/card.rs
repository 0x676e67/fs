use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct CardPredictor(ImageClassifierPredictor);

impl CardPredictor {
    /// Create a new instance of the CardPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("card.onnx", config).await?,
        ))
    }
}

impl Predictor for CardPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
