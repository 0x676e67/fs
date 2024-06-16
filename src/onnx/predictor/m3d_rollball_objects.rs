use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct M3DRotationPredictor(ImagePairClassifierPredictor);

impl M3DRotationPredictor {
    /// Create a new instance of the M3DRotationPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "3d_rollball_objects_v2.onnx",
            config,
            false,
        )?))
    }
}

impl Predictor for M3DRotationPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
