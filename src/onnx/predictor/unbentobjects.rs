use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct UnbentobjectsPredictor(ImageClassifierPredictor);

impl UnbentobjectsPredictor {
    /// Create a new instance of the UnbentobjectsPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "knotsCrossesCircle.onnx",
            config,
        )?))
    }
}

impl Predictor for UnbentobjectsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
