mod base;
#[allow(non_snake_case)]
pub mod brokenJigsawbrokenjigsaw_swap;
pub mod card;
pub mod cardistance;
pub mod conveyor;
pub mod coordinatesmatch;
pub mod counting;
pub mod dice_pair;
pub mod diceico;
pub mod dicematch;
pub mod frankenhead;
pub mod hand_number_puzzle;
pub mod hopscotch_highsec;
pub mod knots_crosses_circle;
pub mod lumber_length_game;
pub mod numericalmatch;
pub mod orbit_match_game;
pub mod penguins;
pub mod penguins_icon;
pub mod rockstack;
pub mod rollball_animals_multi;
pub mod rollball_objects;
pub mod shadows;
pub mod train_coordinates;
pub mod unbentobjects;

pub trait Predictor: Send + Sync {
    fn predict(&self, image: image::DynamicImage) -> crate::Result<i32>;

    fn active(&self) -> bool;
}
