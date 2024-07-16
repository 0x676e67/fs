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
pub mod maze2;
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

use base64::{engine::general_purpose, Engine as _};

pub trait Predictor: Send + Sync {
    fn predict_base64(&self, image: &String) -> crate::Result<i32> {
        let image_bytes =
            general_purpose::STANDARD.decode(image.split(',').nth(1).unwrap_or(image))?;
        let image = image::load_from_memory(&image_bytes)?;
        self.predict(image)
    }

    fn predict(&self, image: image::DynamicImage) -> crate::Result<i32>;

    fn active(&self) -> bool;
}
