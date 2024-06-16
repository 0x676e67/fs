use crate::onnx::ONNXConfig;

use super::{base::ImageClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct KnotsCrossesCirclePredictor(ImageClassifierPredictor);

impl KnotsCrossesCirclePredictor {
    /// Create a new instance of the KnotsCrossesCirclePredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "knotsCrossesCircle.onnx",
            config,
        )?))
    }
}

impl Predictor for KnotsCrossesCirclePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
