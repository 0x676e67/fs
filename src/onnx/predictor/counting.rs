use super::{base::ImageClassifierPredictor, Predictor};
use crate::onnx::ONNXConfig;
use crate::Result;
use image::DynamicImage;
pub struct CountingPredictor(ImageClassifierPredictor);

impl CountingPredictor {
    /// Create a new instance of the CountingPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "counting.onnx",
            config,
        )?))
    }
}

impl Predictor for CountingPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
