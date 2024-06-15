mod base;
#[allow(non_snake_case)]
mod brokenJigsawbrokenjigsaw_swap;
mod card;
mod cardistance;
mod conveyor;
mod coordinatesmatch;
mod counting;
mod dicematch;
mod frankenhead;
mod hand_number_puzzle;
mod hopscotch_highsec;
mod image_processing;
mod knots_crosses_circle;
mod m3d_rollball_objects;
mod numericalmatch;
mod penguin;
mod penguins_icon;
mod rockstack;
mod shadows;
mod train_coordinates;
mod unbentobjects;

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
use conveyor::ConveyorPredictor;
use image::DynamicImage;
use numericalmatch::NumericalmatchPredictor;
use tokio::sync::OnceCell;
use unbentobjects::UnbentobjectsPredictor;

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
static NUMERICALMATCH_PREDICTOR: OnceCell<NumericalmatchPredictor> = OnceCell::const_new();
static CONVEYOR_PREDICTOR: OnceCell<ConveyorPredictor> = OnceCell::const_new();
static UNBENTOBJECTS_PREDICTOR: OnceCell<UnbentobjectsPredictor> = OnceCell::const_new();

/// Predictor trait
pub trait Predictor: Send + Sync {
    fn predict(&self, image: DynamicImage) -> Result<i32>;
}

/// Get the model predictor for the given model type
pub async fn get_predictor(
    model_type: ModelType,
    args: &BootArgs,
) -> Result<&'static dyn Predictor> {
    let predictor = match model_type {
        ModelType::Unbentobjects => {
            get_predictor_from_cell(&UNBENTOBJECTS_PREDICTOR, || {
                UnbentobjectsPredictor::new(args)
            })
            .await?
        }

        ModelType::Conveyor => {
            get_predictor_from_cell(&CONVEYOR_PREDICTOR, || ConveyorPredictor::new(args)).await?
        }
        ModelType::Numericalmatch => {
            get_predictor_from_cell(&NUMERICALMATCH_PREDICTOR, || {
                NumericalmatchPredictor::new(args)
            })
            .await?
        }
        ModelType::M3dRollballAnimals | ModelType::M3dRollballObjects => {
            get_predictor_from_cell(&M3D_ROLLBALL_PREDICTOR, || M3DRotationPredictor::new(args))
                .await?
        }
        ModelType::Coordinatesmatch => {
            get_predictor_from_cell(&COORDINATES_MATCH_PREDICTOR, || {
                CoordinatesMatchPredictor::new(args)
            })
            .await?
        }
        ModelType::HopscotchHighsec => {
            get_predictor_from_cell(&HOPSCOTCH_HIGHSEC_PREDICTOR, || {
                HopscotchHighsecPredictor::new(args)
            })
            .await?
        }
        ModelType::TrainCoordinates => {
            get_predictor_from_cell(&TRAIN_COORDINATES_PREDICTOR, || {
                TrainCoordinatesPredictor::new(args)
            })
            .await?
        }
        ModelType::Penguin => {
            get_predictor_from_cell(&PENGUIN_PREDICTOR, || PenguinPredictor::new(args)).await?
        }
        ModelType::Shadows => {
            get_predictor_from_cell(&SHADOWS_PREDICTOR, || ShadowsPredictor::new(args)).await?
        }
        ModelType::BrokenJigsawbrokenjigsaw_swap => {
            get_predictor_from_cell(&BROKEN_JIGSAW_BROKEN_JIGSAW_SWAPL, || {
                BrokenJigsawbrokenjigsaw_swap::new(args)
            })
            .await?
        }
        ModelType::Frankenhead => {
            get_predictor_from_cell(&FRANKENHEAD_PREDICTOR, || FrankenheadPredictor::new(args))
                .await?
        }
        ModelType::Counting => {
            get_predictor_from_cell(&COUNTING_PREDICTOR, || CountingPredictor::new(args)).await?
        }
        ModelType::Card => {
            get_predictor_from_cell(&CARD_PREDICTOR, || CardPredictor::new(args)).await?
        }
        ModelType::Rockstack => {
            get_predictor_from_cell(&ROCKSTACK_PREDICTOR, || RockstackPredictor::new(args)).await?
        }
        ModelType::Cardistance => {
            get_predictor_from_cell(&CARDISTANCE_PREDICTOR, || CardistancePredictor::new(args))
                .await?
        }
        ModelType::PenguinsIcon => {
            get_predictor_from_cell(&PENGUINS_ICON_PREDICTOR, || {
                PenguinsIconPredictor::new(args)
            })
            .await?
        }
        ModelType::KnotsCrossesCircle => {
            get_predictor_from_cell(&KNOTS_CROSSES_CIRCLE_PREDICTOR, || {
                KnotsCrossesCirclePredictor::new(args)
            })
            .await?
        }
        ModelType::HandNumberPuzzle => {
            get_predictor_from_cell(&HAND_NUMBER_PUZZLE_PREDICTOR, || {
                HandNumberPuzzlePredictor::new(args)
            })
            .await?
        }
        ModelType::Dicematch => {
            get_predictor_from_cell(&DICEMATCH_PREDICTOR, || DicematchMatchPredictor::new(args))
                .await?
        }
    };
    Ok(predictor)
}

async fn get_predictor_from_cell<P, F>(
    cell: &'static OnceCell<P>,
    creator: F,
) -> Result<&'static dyn Predictor>
where
    P: Predictor + 'static,
    F: FnOnce() -> Result<P>,
{
    Ok(cell.get_or_try_init(|| async { creator() }).await? as &'static dyn Predictor)
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
    Numericalmatch,
    Conveyor,
    Unbentobjects,
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
            "numericalmatch" => Ok(ModelType::Numericalmatch),
            "conveyor" => Ok(ModelType::Conveyor),
            "unbentobjects" => Ok(ModelType::Unbentobjects),
            // fallback to M3dRollballObjects
            _ => Err(anyhow::anyhow!("unknown model type")),
        }
    }
}
