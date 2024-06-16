use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct HopscotchHighsecPredictor(ImagePairClassifierPredictor);

impl HopscotchHighsecPredictor {
    /// Create a new instance of the HopscotchHighsecPredictor
    pub fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "hopscotch_highsec.onnx",
            config,
            false,
        )?))
    }
}

impl Predictor for HopscotchHighsecPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
