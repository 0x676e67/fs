use crate::error::Error;
use crate::serve::Task;
use crate::Result;

#[derive(Debug)]
pub enum Variant {
    M3dRollballAnimals,
    M3dRollballObjects,
    Coordinatesmatch,
    HopscotchHighsec,
    TrainCoordinates,
    Penguin,
    Shadows,
    #[allow(non_camel_case_types)]
    BrokenJigsawbrokenjigsaw_swap,
    Frankenhead,
    Counting,
    Card,
    Rockstack,
    Cardistance,
    PenguinsIcon,
    KnotsCrossesCircle,
    HandNumberPuzzle,
    Dicematch,
    Numericalmatch,
    Conveyor,
    Unbentobjects,
    LumberLengthGame,
}

impl TryFrom<&Task> for Variant {
    type Error = Error;
    fn try_from(task: &Task) -> Result<Self> {
        let variant = match task.game_variant_instructions.0.as_str() {
            "3d_rollball_animals" => Variant::M3dRollballAnimals,
            "3d_rollball_objects" => Variant::M3dRollballObjects,
            "coordinatesmatch" => Variant::Coordinatesmatch,
            "hopscotch_highsec" => Variant::HopscotchHighsec,
            "train_coordinates" => Variant::TrainCoordinates,
            "penguin" => Variant::Penguin,
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
            _ => Err(Error::UnknownVariantType(
                task.game_variant_instructions.0.clone(),
            ))?,
        };

        Ok(variant)
    }
}
