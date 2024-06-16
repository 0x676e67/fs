use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct PenguinPredictor(ImageClassifierPredictor);

impl PenguinPredictor {
    /// Create a new instance of the TrainCoordinatesPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new("penguin.onnx", config)?))
    }
}

impl Predictor for PenguinPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
