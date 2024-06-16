use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct PenguinsIconPredictor(ImageClassifierPredictor);

impl PenguinsIconPredictor {
    /// Create a new instance of the PenguinsIconPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "penguins-icon.onnx",
            config,
        )?))
    }
}

impl Predictor for PenguinsIconPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
