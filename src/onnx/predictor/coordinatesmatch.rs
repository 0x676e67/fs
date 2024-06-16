use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

use crate::onnx::ONNXConfig;
pub struct CoordinatesMatchPredictor(ImagePairClassifierPredictor);

impl CoordinatesMatchPredictor {
    /// Create a new instance of the CoordinatesMatchPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "coordinatesmatch.onnx",
            config,
            false,
        )?))
    }
}

impl Predictor for CoordinatesMatchPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
