use std::{fmt, sync::Arc};

use serde::{
    de::{SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};

#[derive(Deserialize)]
pub struct Task {
    /// API key
    pub api_key: Option<String>,
    /// base64 image list, e.g.
    /// ["/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"]
    #[serde(deserialize_with = "deserialize_images")]
    pub images: Vec<Arc<String>>,
    /// image type variant, e.g. ["3d_rollball_objects", "Use the arrows to
    /// rotate the object to face in the direction of the hand"]
    pub game_variant_instructions: (String, String),
}

fn deserialize_images<'de, D>(deserializer: D) -> Result<Vec<Arc<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct VecArcStringVisitor;

    impl<'de> Visitor<'de> for VecArcStringVisitor {
        type Value = Vec<Arc<String>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of strings")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut images = Vec::new();

            while let Some(value) = seq.next_element::<String>()? {
                images.push(Arc::new(value));
            }

            Ok(images)
        }
    }

    deserializer.deserialize_seq(VecArcStringVisitor)
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
