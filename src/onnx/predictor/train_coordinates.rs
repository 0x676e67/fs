use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct TrainCoordinatesPredictor(ImagePairClassifierPredictor);

impl TrainCoordinatesPredictor {
    /// Create a new instance of the TrainCoordinatesPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("train_coordinates.onnx", None, config, false)
                .await?,
        ))
    }
}

impl Predictor for TrainCoordinatesPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
