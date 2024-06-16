use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

pub struct M3DRotationMultiPredictor(ImagePairClassifierPredictor);

impl M3DRotationMultiPredictor {
    /// Create a new instance of the M3DRotationPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("3d_rollball_animals_multi.onnx", config, false)
                .await?,
        ))
    }
}

impl Predictor for M3DRotationMultiPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
