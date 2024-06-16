use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct RockstackPredictor(ImagePairClassifierPredictor);

impl RockstackPredictor {
    /// Create a new instance of the RockstackPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "rockstack_v2.onnx",
            config,
            true,
        )?))
    }
}

impl Predictor for RockstackPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
