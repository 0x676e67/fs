use crate::{error::Error, serve::Task, Result};

#[derive(Debug, Clone, Copy, proc_variant::EnumVariantCount)]
pub enum Variant {
    RollballAnimals = 0,
    RollballObjects = 1,
    RollballAnimalsMulti = 2,
    Coordinatesmatch = 3,
    HopscotchHighsec = 4,
    TrainCoordinates = 5,
    Penguins = 6,
    Shadows = 7,
    #[allow(non_camel_case_types)]
    BrokenJigsawbrokenjigsaw_swap = 8,
    Frankenhead = 9,
    Counting = 10,
    Card = 11,
    Rockstack = 12,
    Cardistance = 13,
    PenguinsIcon = 14,
    KnotsCrossesCircle = 15,
    HandNumberPuzzle = 16,
    Dicematch = 17,
    Numericalmatch = 18,
    Conveyor = 19,
    Unbentobjects = 20,
    LumberLengthGame = 21,
    DicePair = 22,
    OrbitMatchGame = 23,
    Diceico = 24,
}

impl Variant {
    /// Returns the number of variants in the enum.
    pub const fn const_count() -> usize {
        LENGTH
    }
}

impl TryFrom<&Task> for Variant {
    type Error = Error;

    fn try_from(task: &Task) -> Result<Self> {
        let variant = match task.game_variant_instructions.0.as_str() {
            "3d_rollball_animals" => Variant::RollballAnimals,
            "3d_rollball_objects" => Variant::RollballObjects,
            "3d_rollball_animals_multi" => Variant::RollballAnimalsMulti,
            "coordinatesmatch" => Variant::Coordinatesmatch,
            "hopscotch_highsec" => Variant::HopscotchHighsec,
            "train_coordinates" => Variant::TrainCoordinates,
            "penguins" => Variant::Penguins,
            "shadows" => Variant::Shadows,
            "BrokenJigsawbrokenjigsaw_swap" => Variant::BrokenJigsawbrokenjigsaw_swap,
            "frankenhead" => Variant::Frankenhead,
            "counting" => Variant::Counting,
            "card" => Variant::Card,
            "rockstack" => Variant::Rockstack,
            "cardistance" => Variant::Cardistance,
            "penguins-icon" => Variant::PenguinsIcon,
            "knotsCrossesCircle" => Variant::KnotsCrossesCircle,
            "hand_number_puzzle" => Variant::HandNumberPuzzle,
            "dicematch" => Variant::Dicematch,
            "numericalmatch" => Variant::Numericalmatch,
            "conveyor" => Variant::Conveyor,
            "unbentobjects" => Variant::Unbentobjects,
            "lumber-length-game" => Variant::LumberLengthGame,
            "dice_pair" => Variant::DicePair,
            "orbit_match_game" => Variant::OrbitMatchGame,
            "diceico" => Variant::Diceico,
            _ => Err(Error::UnknownVariantType(
                task.game_variant_instructions.0.clone(),
            ))?,
        };

        Ok(variant)
    }
}
