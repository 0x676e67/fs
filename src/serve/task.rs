use image::ImageError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Task {
    /// API key
    pub api_key: Option<String>,
    /// base64 image list, e.g. ["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"]
    pub images: Vec<String>,
    /// image type variant, e.g. ["3d_rollball_objects", "Use the arrows to rotate the object to face in the direction of the hand"]
    pub game_variant_instructions: (String, String),
}

#[derive(Debug, Serialize)]
pub struct TaskResult {
    /// error message, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// whether the model is a solve
    pub solved: bool,
    /// whether the model is a classifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<i32>>,
}

impl From<ImageError> for TaskResult {
    fn from(result: ImageError) -> Self {
        Self {
            error: Some(result.to_string()),
            solved: false,
            objects: None,
        }
    }
}
impl From<base64::DecodeError> for TaskResult {
    fn from(result: base64::DecodeError) -> Self {
        Self {
            error: Some(result.to_string()),
            solved: false,
            objects: None,
        }
    }
}
