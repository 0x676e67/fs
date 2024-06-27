use super::{base::ImageClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct UnbentobjectsPredictor(ImageClassifierPredictor);

impl UnbentobjectsPredictor {
    /// Create a new instance of the UnbentobjectsPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImageClassifierPredictor::new("knotsCrossesCircle.onnx", None, config).await?,
        ))
    }
}

impl Predictor for UnbentobjectsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
