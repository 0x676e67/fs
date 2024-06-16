use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct ConveyorPredictor(ImagePairClassifierPredictor);

impl ConveyorPredictor {
    /// Create a new instance of the ConveyorPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "conveyor.onnx",
            config,
            false,
        )?))
    }
}

impl Predictor for ConveyorPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
