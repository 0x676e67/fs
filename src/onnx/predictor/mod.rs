mod base;
#[allow(non_snake_case)]
pub mod brokenJigsawbrokenjigsaw_swap;
pub mod card;
pub mod cardistance;
pub mod conveyor;
pub mod coordinatesmatch;
pub mod counting;
pub mod dicematch;
pub mod frankenhead;
pub mod hand_number_puzzle;
pub mod hopscotch_highsec;
pub mod knots_crosses_circle;
pub mod lumber_length_game;
pub mod m3d_rollball_objects;
pub mod numericalmatch;
pub mod penguin;
pub mod penguins_icon;
pub mod rockstack;
pub mod shadows;
pub mod train_coordinates;
pub mod unbentobjects;

pub trait Predictor: Send + Sync {
    fn predict(&self, image: image::DynamicImage) -> crate::Result<i32>;
}
