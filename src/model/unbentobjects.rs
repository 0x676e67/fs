use super::{base::ImageClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct UnbentobjectsPredictor(ImageClassifierPredictor);

impl UnbentobjectsPredictor {
    /// Create a new instance of the UnbentobjectsPredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImageClassifierPredictor::new(
            "knotsCrossesCircle.onnx",
            args,
        )?))
    }
}

impl Predictor for UnbentobjectsPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
