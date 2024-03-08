mod base;
#[allow(non_snake_case)]
mod brokenJigsawbrokenjigsaw_swap;
mod card;
mod cardistance;
mod coordinatesmatch;
mod counting;
mod dicematch;
mod frankenhead;
mod hand_number_puzzle;
mod hopscotch_highsec;
mod image_processing;
mod knots_crosses_circle;
mod m3d_rollball_objects;
mod penguin;
mod penguins_icon;
mod rockstack;
mod shadows;
mod train_coordinates;

use std::str::FromStr;

use self::{
    brokenJigsawbrokenjigsaw_swap::BrokenJigsawbrokenjigsaw_swap, card::CardPredictor,
    cardistance::CardistancePredictor, coordinatesmatch::CoordinatesMatchPredictor,
    counting::CountingPredictor, dicematch::DicematchMatchPredictor,
    frankenhead::FrankenheadPredictor, hand_number_puzzle::HandNumberPuzzlePredictor,
    hopscotch_highsec::HopscotchHighsecPredictor,
    knots_crosses_circle::KnotsCrossesCirclePredictor, m3d_rollball_objects::M3DRotationPredictor,
    penguin::PenguinPredictor, penguins_icon::PenguinsIconPredictor, rockstack::RockstackPredictor,
    shadows::ShadowsPredictor, train_coordinates::TrainCoordinatesPredictor,
};
use crate::BootArgs;
use anyhow::Result;
use image::DynamicImage;
use tokio::sync::OnceCell;

static M3D_ROLLBALL_PREDICTOR: OnceCell<M3DRotationPredictor> = OnceCell::const_new();
static COORDINATES_MATCH_PREDICTOR: OnceCell<CoordinatesMatchPredictor> = OnceCell::const_new();
static HOPSCOTCH_HIGHSEC_PREDICTOR: OnceCell<HopscotchHighsecPredictor> = OnceCell::const_new();
static TRAIN_COORDINATES_PREDICTOR: OnceCell<TrainCoordinatesPredictor> = OnceCell::const_new();
static PENGUIN_PREDICTOR: OnceCell<PenguinPredictor> = OnceCell::const_new();
static SHADOWS_PREDICTOR: OnceCell<ShadowsPredictor> = OnceCell::const_new();
static BROKEN_JIGSAW_BROKEN_JIGSAW_SWAPL: OnceCell<BrokenJigsawbrokenjigsaw_swap> =
    OnceCell::const_new();
static FRANKENHEAD_PREDICTOR: OnceCell<FrankenheadPredictor> = OnceCell::const_new();
static COUNTING_PREDICTOR: OnceCell<CountingPredictor> = OnceCell::const_new();
static CARD_PREDICTOR: OnceCell<CardPredictor> = OnceCell::const_new();
static ROCKSTACK_PREDICTOR: OnceCell<RockstackPredictor> = OnceCell::const_new();
static CARDISTANCE_PREDICTOR: OnceCell<CardistancePredictor> = OnceCell::const_new();
static PENGUINS_ICON_PREDICTOR: OnceCell<PenguinsIconPredictor> = OnceCell::const_new();
static KNOTS_CROSSES_CIRCLE_PREDICTOR: OnceCell<KnotsCrossesCirclePredictor> =
    OnceCell::const_new();
static HAND_NUMBER_PUZZLE_PREDICTOR: OnceCell<HandNumberPuzzlePredictor> = OnceCell::const_new();
static DICEMATCH_PREDICTOR: OnceCell<DicematchMatchPredictor> = OnceCell::const_new();

/// Predictor trait
pub trait Predictor: Send + Sync {
    fn predict(&self, image: DynamicImage) -> Result<i32>;
}

/// Load the models predictor
pub fn init_predictor(args: &BootArgs) -> Result<()> {
    set_predictor(&M3D_ROLLBALL_PREDICTOR, || M3DRotationPredictor::new(args))?;
    set_predictor(&COORDINATES_MATCH_PREDICTOR, || {
        CoordinatesMatchPredictor::new(args)
    })?;
    set_predictor(&HOPSCOTCH_HIGHSEC_PREDICTOR, || {
        HopscotchHighsecPredictor::new(args)
    })?;
    set_predictor(&TRAIN_COORDINATES_PREDICTOR, || {
        TrainCoordinatesPredictor::new(args)
    })?;
    set_predictor(&PENGUIN_PREDICTOR, || PenguinPredictor::new(args))?;
    set_predictor(&SHADOWS_PREDICTOR, || ShadowsPredictor::new(args))?;
    set_predictor(&BROKEN_JIGSAW_BROKEN_JIGSAW_SWAPL, || {
        BrokenJigsawbrokenjigsaw_swap::new(args)
    })?;
    set_predictor(&FRANKENHEAD_PREDICTOR, || FrankenheadPredictor::new(args))?;
    set_predictor(&COUNTING_PREDICTOR, || CountingPredictor::new(args))?;
    set_predictor(&CARD_PREDICTOR, || CardPredictor::new(args))?;
    set_predictor(&ROCKSTACK_PREDICTOR, || RockstackPredictor::new(args))?;
    set_predictor(&CARDISTANCE_PREDICTOR, || CardistancePredictor::new(args))?;
    set_predictor(&PENGUINS_ICON_PREDICTOR, || {
        PenguinsIconPredictor::new(args)
    })?;
    set_predictor(&KNOTS_CROSSES_CIRCLE_PREDICTOR, || {
        KnotsCrossesCirclePredictor::new(args)
    })?;
    set_predictor(&HAND_NUMBER_PUZZLE_PREDICTOR, || {
        HandNumberPuzzlePredictor::new(args)
    })?;
    set_predictor(&DICEMATCH_PREDICTOR, || DicematchMatchPredictor::new(args))?;
    Ok(())
}

/// Get the model predictor for the given model type
pub fn get_predictor(model_type: ModelType) -> Result<&'static dyn Predictor> {
    let predictor = match model_type {
        ModelType::M3dRollballAnimals | ModelType::M3dRollballObjects => {
            get_predictor_from_cell(&M3D_ROLLBALL_PREDICTOR)?
        }
        ModelType::Coordinatesmatch => get_predictor_from_cell(&COORDINATES_MATCH_PREDICTOR)?,
        ModelType::HopscotchHighsec => get_predictor_from_cell(&HOPSCOTCH_HIGHSEC_PREDICTOR)?,
        ModelType::TrainCoordinates => get_predictor_from_cell(&TRAIN_COORDINATES_PREDICTOR)?,
        ModelType::Penguin => get_predictor_from_cell(&PENGUIN_PREDICTOR)?,
        ModelType::Shadows => get_predictor_from_cell(&SHADOWS_PREDICTOR)?,
        ModelType::BrokenJigsawbrokenjigsaw_swap => {
            get_predictor_from_cell(&BROKEN_JIGSAW_BROKEN_JIGSAW_SWAPL)?
        }
        ModelType::Frankenhead => get_predictor_from_cell(&FRANKENHEAD_PREDICTOR)?,
        ModelType::Counting => get_predictor_from_cell(&COUNTING_PREDICTOR)?,
        ModelType::Card => get_predictor_from_cell(&CARD_PREDICTOR)?,
        ModelType::Rockstack => get_predictor_from_cell(&ROCKSTACK_PREDICTOR)?,
        ModelType::Cardistance => get_predictor_from_cell(&CARDISTANCE_PREDICTOR)?,
        ModelType::PenguinsIcon => get_predictor_from_cell(&PENGUINS_ICON_PREDICTOR)?,
        ModelType::KnotsCrossesCircle => get_predictor_from_cell(&KNOTS_CROSSES_CIRCLE_PREDICTOR)?,
        ModelType::HandNumberPuzzle => get_predictor_from_cell(&HAND_NUMBER_PUZZLE_PREDICTOR)?,
        ModelType::Dicematch => get_predictor_from_cell(&DICEMATCH_PREDICTOR)?,
    };
    Ok(predictor)
}

fn set_predictor<P, F>(cell: &OnceCell<P>, creator: F) -> Result<()>
where
    P: Predictor,
    F: FnOnce() -> Result<P>,
{
    cell.set(creator()?)
        .map_err(|_| anyhow::anyhow!("failed to load models"))
}

fn get_predictor_from_cell<P>(cell: &'static OnceCell<P>) -> Result<&'static dyn Predictor>
where
    P: Predictor + 'static,
{
    let predictor = cell
        .get()
        .ok_or_else(|| anyhow::anyhow!("models not loaded"))?;
    Ok(predictor as &'static dyn Predictor)
}

#[derive(Debug)]
pub enum ModelType {
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
}

impl FromStr for ModelType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "3d_rollball_animals" => Ok(ModelType::M3dRollballAnimals),
            "3d_rollball_objects" => Ok(ModelType::M3dRollballObjects),
            "coordinatesmatch" => Ok(ModelType::Coordinatesmatch),
            "hopscotch_highsec" => Ok(ModelType::HopscotchHighsec),
            "train_coordinates" => Ok(ModelType::TrainCoordinates),
            "penguin" => Ok(ModelType::Penguin),
            "shadows" => Ok(ModelType::Shadows),
            "BrokenJigsawbrokenjigsaw_swap" => Ok(ModelType::BrokenJigsawbrokenjigsaw_swap),
            "frankenhead" => Ok(ModelType::Frankenhead),
            "counting" => Ok(ModelType::Counting),
            "card" => Ok(ModelType::Card),
            "rockstack" => Ok(ModelType::Rockstack),
            "cardistance" => Ok(ModelType::Cardistance),
            "penguins-icon" => Ok(ModelType::PenguinsIcon),
            "knotsCrossesCircle" => Ok(ModelType::KnotsCrossesCircle),
            "hand_number_puzzle" => Ok(ModelType::HandNumberPuzzle),
            "dicematch" => Ok(ModelType::Dicematch),
            // fallback to M3dRollballObjects
            _ => Err(anyhow::anyhow!("unknown model type")),
        }
    }
}
