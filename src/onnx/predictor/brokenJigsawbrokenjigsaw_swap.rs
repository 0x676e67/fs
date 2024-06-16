use crate::onnx::ONNXConfig;

use super::{base::ImagePairClassifierPredictor, Predictor};
use crate::Result;
use image::DynamicImage;

#[allow(non_camel_case_types)]
pub struct BrokenJigsawbrokenjigsaw_swap(ImagePairClassifierPredictor);

impl BrokenJigsawbrokenjigsaw_swap {
    /// Create a new instance of the BrokenJigsawbrokenjigsaw_swapl
    pub async fn new(args: &ONNXConfig) -> Result<Self> {
        Ok(Self(
            ImagePairClassifierPredictor::new("BrokenJigsawbrokenjigsaw_swap.onnx", args, false)
                .await?,
        ))
    }
}

impl Predictor for BrokenJigsawbrokenjigsaw_swap {
    fn predict(&self, image: DynamicImage) -> Result<i32> {
        self.0.predict(image)
    }
}
