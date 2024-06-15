use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;

pub struct ConveyorPredictor(ImagePairClassifierPredictor);

impl ConveyorPredictor {
    /// Create a new instance of the ConveyorPredictor
    pub fn new(args: &BootArgs) -> Result<Self> {
        Ok(Self(ImagePairClassifierPredictor::new(
            "conveyor.onnx",
            args,
            false,
        )?))
    }
}

impl Predictor for ConveyorPredictor {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
