use crate::error::Error;
use crate::Result;
use std::str::FromStr;

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

impl FromStr for Variant {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "3d_rollball_animals" => Ok(Variant::M3dRollballAnimals),
            "3d_rollball_objects" => Ok(Variant::M3dRollballObjects),
            "coordinatesmatch" => Ok(Variant::Coordinatesmatch),
            "hopscotch_highsec" => Ok(Variant::HopscotchHighsec),
            "train_coordinates" => Ok(Variant::TrainCoordinates),
            "penguin" => Ok(Variant::Penguin),
            "shadows" => Ok(Variant::Shadows),
            "BrokenJigsawbrokenjigsaw_swap" => Ok(Variant::BrokenJigsawbrokenjigsaw_swap),
            "frankenhead" => Ok(Variant::Frankenhead),
            "counting" => Ok(Variant::Counting),
            "card" => Ok(Variant::Card),
            "rockstack" => Ok(Variant::Rockstack),
            "cardistance" => Ok(Variant::Cardistance),
            "penguins-icon" => Ok(Variant::PenguinsIcon),
            "knotsCrossesCircle" => Ok(Variant::KnotsCrossesCircle),
            "hand_number_puzzle" => Ok(Variant::HandNumberPuzzle),
            "dicematch" => Ok(Variant::Dicematch),
            "numericalmatch" => Ok(Variant::Numericalmatch),
            "conveyor" => Ok(Variant::Conveyor),
            "unbentobjects" => Ok(Variant::Unbentobjects),
            "lumber-length-game" => Ok(Variant::LumberLengthGame),
            _ => Err(Error::UnknownVariantType(s.to_string())),
        }
    }
}
