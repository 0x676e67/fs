use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct Numericalmatch(ImagePairClassifierPredictor);

impl Numericalmatch {
    /// Create a new instance of the Numericalmatch
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "numericalmatch.onnx",
            args,
            false,
        )?))
    }
}

impl Predictor for Numericalmatch {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
