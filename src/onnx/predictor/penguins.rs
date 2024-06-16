use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct PenguinsPredictor(ImageClassifierPredictor);

impl PenguinsPredictor {
    /// Create a new instance of the PenguinsPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new("penguin.onnx", config)?))
    }
}

impl Predictor for PenguinsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
