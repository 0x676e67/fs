mod predictor;
mod util;
mod variant;

use crate::Result;
pub use predictor::Predictor;
use predictor::{
    brokenJigsawbrokenjigsaw_swap::BrokenJigsawbrokenjigsaw_swap, card::CardPredictor,
    cardistance::CardistancePredictor, conveyor::ConveyorPredictor,
    coordinatesmatch::CoordinatesMatchPredictor, counting::CountingPredictor,
    dice_pair::DicePairPredictor, dicematch::DicematchMatchPredictor,
    frankenhead::FrankenheadPredictor, hand_number_puzzle::HandNumberPuzzlePredictor,
    hopscotch_highsec::HopscotchHighsecPredictor,
    knots_crosses_circle::KnotsCrossesCirclePredictor,
    lumber_length_game::LumberLengthGamePredictor, m3d_rollball_objects::M3DRotationPredictor,
    numericalmatch::NumericalmatchPredictor, penguins::PenguinsPredictor,
    penguins_icon::PenguinsIconPredictor, rockstack::RockstackPredictor, shadows::ShadowsPredictor,
    train_coordinates::TrainCoordinatesPredictor, unbentobjects::UnbentobjectsPredictor,
};
use std::{future::Future, path::PathBuf};
use tokio::sync::OnceCell;
pub use variant::Variant;

static M3D_ROLLBALL_PREDICTOR: OnceCell<M3DRotationPredictor> = OnceCell::const_new();
static COORDINATES_MATCH_PREDICTOR: OnceCell<CoordinatesMatchPredictor> = OnceCell::const_new();
static HOPSCOTCH_HIGHSEC_PREDICTOR: OnceCell<HopscotchHighsecPredictor> = OnceCell::const_new();
static TRAIN_COORDINATES_PREDICTOR: OnceCell<TrainCoordinatesPredictor> = OnceCell::const_new();
static PENGUIN_PREDICTOR: OnceCell<PenguinsPredictor> = OnceCell::const_new();
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
static LUMBER_LENGTH_GAME_PREDICTOR: OnceCell<LumberLengthGamePredictor> = OnceCell::const_new();
static DICE_PAIR_PREDICTOR: OnceCell<DicePairPredictor> = OnceCell::const_new();

#[derive(typed_builder::TypedBuilder)]
pub struct ONNXConfig {
    model_dir: Option<PathBuf>,
    update_check: bool,
    num_threads: u16,
    allocator: ort::AllocatorType,
}

impl Default for ONNXConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            update_check: false,
            num_threads: 4,
            allocator: ort::AllocatorType::Arena,
        }
    }
}

/// Get the model predictor for the given variant
pub async fn get_predictor(
    variant: Variant,
    config: &ONNXConfig,
) -> Result<&'static dyn Predictor> {
    match variant {
        Variant::DicePair => {
            get_predictor_from_cell(&DICE_PAIR_PREDICTOR, || DicePairPredictor::new(config)).await
        }
        Variant::LumberLengthGame => {
            get_predictor_from_cell(&LUMBER_LENGTH_GAME_PREDICTOR, || {
                LumberLengthGamePredictor::new(config)
            })
            .await
        }
        Variant::Unbentobjects => {
            get_predictor_from_cell(&UNBENTOBJECTS_PREDICTOR, || {
                UnbentobjectsPredictor::new(config)
            })
            .await
        }
        Variant::Conveyor => {
            get_predictor_from_cell(&CONVEYOR_PREDICTOR, || ConveyorPredictor::new(config)).await
        }
        Variant::Numericalmatch => {
            get_predictor_from_cell(&NUMERICALMATCH_PREDICTOR, || {
                NumericalmatchPredictor::new(config)
            })
            .await
        }
        Variant::M3dRollballAnimals | Variant::M3dRollballObjects => {
            get_predictor_from_cell(&M3D_ROLLBALL_PREDICTOR, || {
                M3DRotationPredictor::new(config)
            })
            .await
        }
        Variant::Coordinatesmatch => {
            get_predictor_from_cell(&COORDINATES_MATCH_PREDICTOR, || {
                CoordinatesMatchPredictor::new(config)
            })
            .await
        }
        Variant::HopscotchHighsec => {
            get_predictor_from_cell(&HOPSCOTCH_HIGHSEC_PREDICTOR, || {
                HopscotchHighsecPredictor::new(config)
            })
            .await
        }
        Variant::TrainCoordinates => {
            get_predictor_from_cell(&TRAIN_COORDINATES_PREDICTOR, || {
                TrainCoordinatesPredictor::new(config)
            })
            .await
        }
        Variant::Penguins => {
            get_predictor_from_cell(&PENGUIN_PREDICTOR, || PenguinsPredictor::new(config)).await
        }
        Variant::Shadows => {
            get_predictor_from_cell(&SHADOWS_PREDICTOR, || ShadowsPredictor::new(config)).await
        }
        Variant::BrokenJigsawbrokenjigsaw_swap => {
            get_predictor_from_cell(&BROKEN_JIGSAW_BROKEN_JIGSAW_SWAPL, || {
                BrokenJigsawbrokenjigsaw_swap::new(config)
            })
            .await
        }
        Variant::Frankenhead => {
            get_predictor_from_cell(&FRANKENHEAD_PREDICTOR, || FrankenheadPredictor::new(config))
                .await
        }
        Variant::Counting => {
            get_predictor_from_cell(&COUNTING_PREDICTOR, || CountingPredictor::new(config)).await
        }
        Variant::Card => {
            get_predictor_from_cell(&CARD_PREDICTOR, || CardPredictor::new(config)).await
        }
        Variant::Rockstack => {
            get_predictor_from_cell(&ROCKSTACK_PREDICTOR, || RockstackPredictor::new(config)).await
        }
        Variant::Cardistance => {
            get_predictor_from_cell(&CARDISTANCE_PREDICTOR, || CardistancePredictor::new(config))
                .await
        }
        Variant::PenguinsIcon => {
            get_predictor_from_cell(&PENGUINS_ICON_PREDICTOR, || {
                PenguinsIconPredictor::new(config)
            })
            .await
        }
        Variant::KnotsCrossesCircle => {
            get_predictor_from_cell(&KNOTS_CROSSES_CIRCLE_PREDICTOR, || {
                KnotsCrossesCirclePredictor::new(config)
            })
            .await
        }
        Variant::HandNumberPuzzle => {
            get_predictor_from_cell(&HAND_NUMBER_PUZZLE_PREDICTOR, || {
                HandNumberPuzzlePredictor::new(config)
            })
            .await
        }
        Variant::Dicematch => {
            get_predictor_from_cell(&DICEMATCH_PREDICTOR, || {
                DicematchMatchPredictor::new(config)
            })
            .await
        }
    }
}

async fn get_predictor_from_cell<P, F, Fut>(
    cell: &'static OnceCell<P>,
    creator: F,
) -> Result<&'static dyn Predictor>
where
    P: Predictor + 'static,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<P>>,
{
    Ok(cell.get_or_try_init(creator).await? as &'static dyn Predictor)
}
