use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::{onnx::ONNXConfig, Result};
use image::DynamicImage;

pub struct DiceicoPredictor(ImagePairClassifierPredictor);

impl DiceicoPredictor {
    /// Create a new instance of the DiceicoPredictor
    pub async fn new(config: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("diceico.onnx", config, false).await?,
        ))
    }
}

impl Predictor for DiceicoPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }

    fn active(&self) -> bool {
        self.0.active()
    }
}
