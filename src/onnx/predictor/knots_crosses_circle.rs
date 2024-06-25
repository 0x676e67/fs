use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct KnotsCrossesCirclePredictor(ImageClassifierPredictor);

impl KnotsCrossesCirclePredictor {
    /// Create a new instance of the KnotsCrossesCirclePredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("knotsCrossesCircle.onnx", config).await?,
        ))
    }
}

impl Predictor for KnotsCrossesCirclePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
