use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct ConveyorPredictor(ImagePairClassifierPredictor);

impl ConveyorPredictor {
    /// Create a new instance of the ConveyorPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("conveyor.onnx", None, config, false).await?,
        ))
    }
}

impl Predictor for ConveyorPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
