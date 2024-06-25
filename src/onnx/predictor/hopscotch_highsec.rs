use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct HopscotchHighsecPredictor(ImagePairClassifierPredictor);

impl HopscotchHighsecPredictor {
    /// Create a new instance of the HopscotchHighsecPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("hopscotch_highsec.onnx", config, false).await?,
        ))
    }
}

impl Predictor for HopscotchHighsecPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
