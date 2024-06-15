use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct LumberLengthGamePredictor(ImageClassifierPredictor);

impl LumberLengthGamePredictor {
    /// Create a new instance of the LumberLengthGamePredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "lumber-length-game.onnx",
            args,
        )?))
    }
}

impl Predictor for LumberLengthGamePredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
