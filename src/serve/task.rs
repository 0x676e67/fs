use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Task {
    /// API key
    pub api_key: Option<String>,
    /// base64 image list, e.g. ["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"]
    pub images: Vec<String>,
    /// image type variant, e.g. ["3d_rollball_objects", "Use the arrows to rotate the object to face in the direction of the hand"]
    pub game_variant_instructions: (String, String),
}

#[derive(Serialize, typed_builder::TypedBuilder)]
pub struct TaskResult {
    /// error message, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    error: Option<String>,
    /// whether the model is a solve
    solved: bool,
    /// whether the model is a classifier
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    objects: Option<Vec<i32>>,
}
