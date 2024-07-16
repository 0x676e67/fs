mod adapter;
mod predictor;
mod util;
mod variant;

use crate::Result;
pub use adapter::Adapter;
pub use adapter::Config;
pub use predictor::Predictor;
use predictor::{
    brokenJigsawbrokenjigsaw_swap::BrokenJigsawbrokenjigsaw_swap, card::CardPredictor,
    cardistance::CardistancePredictor, conveyor::ConveyorPredictor,
    coordinatesmatch::CoordinatesMatchPredictor, counting::CountingPredictor,
    dice_pair::DicePairPredictor, diceico::DiceicoPredictor, dicematch::DicematchMatchPredictor,
    frankenhead::FrankenheadPredictor, hand_number_puzzle::HandNumberPuzzlePredictor,
    hopscotch_highsec::HopscotchHighsecPredictor,
    knots_crosses_circle::KnotsCrossesCirclePredictor,
    lumber_length_game::LumberLengthGamePredictor, maze2::Maze2Predictor,
    numericalmatch::NumericalmatchPredictor, orbit_match_game::OrbitMatchGamePredictor,
    penguins::PenguinsPredictor, penguins_icon::PenguinsIconPredictor,
    rockstack::RockstackPredictor, rollball_animals_multi::M3DRotationMultiPredictor,
    rollball_objects::M3DRotationPredictor, shadows::ShadowsPredictor,
    train_coordinates::TrainCoordinatesPredictor, unbentobjects::UnbentobjectsPredictor,
};
use std::sync::Arc;
use std::{future::Future, path::PathBuf};
pub use variant::Variant;

#[derive(typed_builder::TypedBuilder)]
pub struct ONNXConfig {
    model_dir: Option<PathBuf>,
    update_check: bool,
    num_threads: u16,
    allocator: ort::AllocatorType,
    onnx_store: Adapter,
}

impl Default for ONNXConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            update_check: false,
            num_threads: 4,
            allocator: ort::AllocatorType::Arena,
            onnx_store: Default::default(),
        }
    }
}

/// Creates a new model predictor based on the provided variant and configuration.
///
/// This function takes a variant and a reference to an ONNXConfig as parameters, and returns a Result
/// that contains a Boxed dynamic Predictor. The specific type of the Predictor depends on the variant.
///
/// # Parameters
/// * `variant`: The variant of the game for which to create a predictor.
/// * `config`: A reference to the ONNXConfig that specifies the configuration for the predictor.
///
/// # Returns
/// A Result that contains a Boxed dynamic Predictor if the predictor was successfully created, or an Error if it was not.
///
/// # Examples
/// ```
/// let config = ONNXConfig::default();
/// let predictor = new_predictor(Variant::OrbitMatchGame, &config).await?;
/// ```
pub async fn new_predictor(variant: Variant, config: &ONNXConfig) -> Result<Arc<dyn Predictor>> {
    match variant {
        Variant::Diceico => get_predictor_from_cell(|| DiceicoPredictor::new(config)).await,
        Variant::OrbitMatchGame => {
            get_predictor_from_cell(|| OrbitMatchGamePredictor::new(config)).await
        }
        Variant::RollballAnimalsMulti => {
            get_predictor_from_cell(|| M3DRotationMultiPredictor::new(config)).await
        }
        Variant::DicePair => get_predictor_from_cell(|| DicePairPredictor::new(config)).await,
        Variant::LumberLengthGame => {
            get_predictor_from_cell(|| LumberLengthGamePredictor::new(config)).await
        }
        Variant::Unbentobjects => {
            get_predictor_from_cell(|| UnbentobjectsPredictor::new(config)).await
        }
        Variant::Conveyor => get_predictor_from_cell(|| ConveyorPredictor::new(config)).await,
        Variant::Numericalmatch => {
            get_predictor_from_cell(|| NumericalmatchPredictor::new(config)).await
        }
        Variant::RollballAnimals | Variant::RollballObjects => {
            get_predictor_from_cell(|| M3DRotationPredictor::new(config)).await
        }
        Variant::Coordinatesmatch => {
            get_predictor_from_cell(|| CoordinatesMatchPredictor::new(config)).await
        }
        Variant::HopscotchHighsec => {
            get_predictor_from_cell(|| HopscotchHighsecPredictor::new(config)).await
        }
        Variant::TrainCoordinates => {
            get_predictor_from_cell(|| TrainCoordinatesPredictor::new(config)).await
        }
        Variant::Penguins => get_predictor_from_cell(|| PenguinsPredictor::new(config)).await,
        Variant::Shadows => get_predictor_from_cell(|| ShadowsPredictor::new(config)).await,
        Variant::BrokenJigsawbrokenjigsaw_swap => {
            get_predictor_from_cell(|| BrokenJigsawbrokenjigsaw_swap::new(config)).await
        }
        Variant::Frankenhead => get_predictor_from_cell(|| FrankenheadPredictor::new(config)).await,
        Variant::Counting => get_predictor_from_cell(|| CountingPredictor::new(config)).await,
        Variant::Card => get_predictor_from_cell(|| CardPredictor::new(config)).await,
        Variant::Rockstack => get_predictor_from_cell(|| RockstackPredictor::new(config)).await,
        Variant::Cardistance => get_predictor_from_cell(|| CardistancePredictor::new(config)).await,
        Variant::PenguinsIcon => {
            get_predictor_from_cell(|| PenguinsIconPredictor::new(config)).await
        }
        Variant::KnotsCrossesCircle => {
            get_predictor_from_cell(|| KnotsCrossesCirclePredictor::new(config)).await
        }
        Variant::HandNumberPuzzle => {
            get_predictor_from_cell(|| HandNumberPuzzlePredictor::new(config)).await
        }
        Variant::Dicematch => {
            get_predictor_from_cell(|| DicematchMatchPredictor::new(config)).await
        }
        Variant::Maze2 => get_predictor_from_cell(|| Maze2Predictor::new(config)).await,
    }
}

async fn get_predictor_from_cell<P, F, Fut>(creator: F) -> Result<Arc<dyn Predictor>>
where
    P: Predictor + 'static,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<P>>,
{
    Ok(Arc::new(creator().await?))
}
