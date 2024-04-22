mod solver;
mod task;

use std::convert::Infallible;
use std::str::FromStr;

use self::solver::Solver;
use self::task::{Task, TaskResult};
use crate::model::{ModelType, Predictor};
use crate::serve::solver::SolverType;
use crate::{model, BootArgs};
use anyhow::Result;
use image::DynamicImage;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use reqwest::StatusCode;
use tokio::sync::OnceCell;
use warp::filters::body::BodyDeserializeError;
use warp::reject::{Reject, Rejection};
use warp::reply::{Json, Reply};
use warp::Filter;

static ARGS: OnceCell<BootArgs> = OnceCell::const_new();
static API_KEY: OnceCell<Option<String>> = OnceCell::const_new();
static SUBMIT_LIMIT: OnceCell<Option<usize>> = OnceCell::const_new();
static SOLVER: OnceCell<Solver> = OnceCell::const_new();

pub struct Serve(BootArgs);

impl Serve {
    pub fn new(args: BootArgs) -> Self {
        Self(args)
    }

    #[tokio::main]
    pub async fn run(self) -> Result<()> {
        // Init args
        ARGS.set(self.0.clone())?;

        // Init API key
        API_KEY.set(self.0.api_key)?;

        // Init submit limit
        SUBMIT_LIMIT.set(Some(self.0.multi_image_limit))?;

        // Init fallback solver
        if let (Some(solver), Some(key)) = (self.0.fallback_solver, self.0.fallback_key) {
            let _ = SOLVER.set(Solver::new(
                SolverType::from_str(&solver)?,
                key,
                self.0.fallback_endpoint,
                self.0.fallback_image_limit,
            ));
        }

        // Init routes
        let routes = warp::path("task")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(handle_task)
            .recover(handle_rejection)
            .with(warp::trace::request());

        tracing::info!("Listening on {}", self.0.bind);

        // Start the server
        match (self.0.tls_cert, self.0.tls_key) {
            (Some(cert), Some(key)) => {
                warp::serve(routes)
                    .tls()
                    .cert_path(cert)
                    .key_path(key)
                    .bind_with_graceful_shutdown(self.0.bind, async {
                        tokio::signal::ctrl_c()
                            .await
                            .expect("failed to install CTRL+C signal handler");
                    })
                    .1
                    .await;
            }
            _ => {
                warp::serve(routes)
                    .bind_with_graceful_shutdown(self.0.bind, async {
                        tokio::signal::ctrl_c()
                            .await
                            .expect("failed to install CTRL+C signal handler");
                    })
                    .1
                    .await;
            }
        }
        Ok(())
    }
}

/// Handle the task
async fn handle_task(task: Task) -> Result<impl Reply, Rejection> {
    // Check the API key
    check_api_key(task.api_key.as_deref()).await?;
    // Check the submit limit
    check_submit_limit(task.images.len()).await?;

    // If the model type is valid, use fallback solver
    if let Ok(model) = ModelType::from_str(task.game_variant_instructions.0.as_str()) {
        // Get the args
        let args = ARGS.get().ok_or_else(|| {
            warp::reject::custom(InternalServerError("args is not initialized".to_owned()))
        })?;
        // handle the solver task
        handle_solver_task(args, task, model).await
    } else {
        // handle the fallback solver task
        handle_fallback_solver_task(task).await
    }
}

/// Handle the model task
async fn handle_solver_task(
    args: &BootArgs,
    task: Task,
    model: ModelType,
) -> Result<Json, Rejection> {
    let predictor = model::get_predictor(model, args)
        .await
        .map_err(|e| warp::reject::custom(BadRequest(e.to_string())))?;

    let objects = if task.images.len() == 1 {
        handle_single_image_task(&task.images[0], predictor)?
    } else {
        handle_multiple_images_task(task.images, predictor)?
    };

    let result = TaskResult {
        error: None,
        solved: true,
        objects,
    };
    Ok(warp::reply::json(&result))
}

/// Handle the single image task
fn handle_single_image_task<P: Predictor + ?Sized>(
    image: &String,
    predictor: &P,
) -> Result<Vec<i32>, Rejection> {
    let image = decode_image(image).map_err(|e| warp::reject::custom(BadRequest(e.to_string())))?;
    let answer = predictor
        .predict(image)
        .map_err(|e| warp::reject::custom(BadRequest(e.to_string())))?;

    Ok(vec![answer])
}

/// Handle the multiple images task
fn handle_multiple_images_task<P: Predictor + ?Sized>(
    images: Vec<String>,
    predictor: &P,
) -> Result<Vec<i32>, Rejection> {
    let mut objects = images
        .into_par_iter()
        .enumerate()
        .map(|(index, image)| {
            let image = decode_image(&image)?;
            let answer = predictor.predict(image)?;
            Ok((index, answer))
        })
        .collect::<Result<Vec<(usize, i32)>>>()
        .map_err(|e| warp::reject::custom(BadRequest(e.to_string())))?;

    objects.sort_by_key(|&(index, _)| index);
    Ok(objects.into_iter().map(|(_, answer)| answer).collect())
}

/// Handle the fallback solver task
async fn handle_fallback_solver_task(task: Task) -> Result<Json, Rejection> {
    let solver = SOLVER.get().ok_or_else(|| {
        warp::reject::custom(InternalServerError("solver is not initialized".to_owned()))
    })?;

    if task.images.len() <= 0 {
        return Err(warp::reject::custom(BadRequest("No images".to_owned())));
    }

    let (game_variant, instructions) = task.game_variant_instructions;

    let mut answers = Vec::with_capacity(task.images.len());
    match solver.solver() {
        solver::SolverType::Yescaptcha => {
            // single image
            for image in task.images {
                // submit task
                let answer = solver
                    .submit_task(solver::SubmitTask {
                        image: Some(&image),
                        images: None,
                        game_variant_instructions: (&game_variant, &instructions),
                    })
                    .await
                    .map_err(|e| warp::reject::custom(InternalServerError(e.to_string())))?;
                answers.extend(answer);
            }
        }
        solver::SolverType::Capsolver => {
            // split chunk images
            let images_chunk = task.images.chunks(solver.limit().max(1));

            // submit multiple images task
            for chunk in images_chunk {
                // submit task
                let answer = solver
                    .submit_task(solver::SubmitTask {
                        image: None,
                        images: Some(chunk),
                        game_variant_instructions: (&game_variant, &instructions),
                    })
                    .await
                    .map_err(|e| warp::reject::custom(InternalServerError(e.to_string())))?;
                answers.extend(answer);
            }
        }
    }

    let result = TaskResult {
        error: None,
        solved: true,
        objects: answers,
    };

    Ok(warp::reply::json(&result))
}

/// Check the API key
async fn check_api_key(api_key: Option<&str>) -> Result<(), Rejection> {
    if let Some(Some(key)) = API_KEY.get() {
        if let Some(api_key) = api_key {
            if key.ne(&api_key) {
                return Err(warp::reject::custom(InvalidTApiKeyError));
            }
        } else {
            return Err(warp::reject::custom(InvalidTApiKeyError));
        }
    }
    Ok(())
}

/// Check the submit limit
async fn check_submit_limit(len: usize) -> Result<(), Rejection> {
    if let Some(Some(limit)) = SUBMIT_LIMIT.get() {
        if len > *limit {
            return Err(warp::reject::custom(InvalidSubmitLimitError));
        }
    }
    Ok(())
}

/// Decode the base64 image
fn decode_image(base64_string: &String) -> Result<DynamicImage> {
    // base64 decode the image
    use base64::{engine::general_purpose, Engine as _};
    let image_bytes = general_purpose::STANDARD
        .decode(base64_string.split(',').nth(1).unwrap_or(base64_string))?;
    // convert the bytes to an image
    Ok(image::load_from_memory(&image_bytes)?)
}

#[derive(Debug)]
struct BadRequest(String);

#[derive(Debug)]
struct InternalServerError(String);

#[derive(Debug)]
struct InvalidTApiKeyError;

#[derive(Debug)]
struct InvalidSubmitLimitError;

impl Reject for BadRequest {}

impl Reject for InternalServerError {}

impl Reject for InvalidTApiKeyError {}

impl Reject for InvalidSubmitLimitError {}

impl Reject for TaskResult {}

async fn handle_rejection(err: Rejection) -> Result<impl Reply, Infallible> {
    let code;
    let message;

    if err.is_not_found() {
        code = StatusCode::NOT_FOUND;
        message = "Not Found".to_owned();
    } else if let Some(e) = err.find::<BadRequest>() {
        code = StatusCode::BAD_REQUEST;
        message = e.0.to_owned();
    } else if err.find::<InvalidTApiKeyError>().is_some() {
        code = StatusCode::UNAUTHORIZED;
        message = "Invalid API key".to_owned();
    } else if let Some(e) = err.find::<BodyDeserializeError>() {
        code = StatusCode::BAD_REQUEST;
        message = e.to_string();
    } else if err.find::<InvalidSubmitLimitError>().is_some() {
        code = StatusCode::BAD_REQUEST;
        if let Some(limit) = SUBMIT_LIMIT.get() {
            message = format!("Invalid submit limit: {}", limit.unwrap_or(0));
        } else {
            message = "Invalid submit limit".to_owned();
        }
    } else if let Some(msg) = err.find::<InternalServerError>() {
        code = StatusCode::INTERNAL_SERVER_ERROR;
        message = msg.0.to_owned();
    } else {
        tracing::info!("Unhandled application error: {:?}", err);
        code = StatusCode::INTERNAL_SERVER_ERROR;
        message = "Internal Server Error".to_owned();
    }

    let json = warp::reply::json(&TaskResult {
        error: Some(message),
        solved: false,
        objects: vec![],
    });

    Ok(warp::reply::with_status(json, code))
}
