mod signal;
mod solver;
mod task;

use std::str::FromStr;
use std::sync::Arc;

use self::solver::Solver;
pub use self::task::Task;
use crate::error::Error;
use crate::onnx::Variant;
use crate::onnx::{ONNXConfig, Predictor};
use crate::serve::solver::SolverType;
use crate::Result;
use crate::{onnx, BootArgs};
use axum::extract::State;
use axum::response::{Html, IntoResponse};
use axum::routing::post;
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use image::DynamicImage;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
pub use task::TaskResult;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnResponse, TraceLayer};
use tracing::Level;

/// Application state
struct AppState {
    // API key
    api_key: Option<String>,
    // Submit image limit
    limit: usize,
    // Fallback solver
    fallback_solver: Option<Solver>,
    // ONNX configuration
    onnx: onnx::ONNXConfig,
}

#[tokio::main]
pub async fn run(args: BootArgs) -> Result<()> {
    // Initialize the logger.
    tracing_subscriber::fmt()
        .with_max_level(if args.debug {
            Level::DEBUG
        } else {
            Level::INFO
        })
        .init();

    // Initialize the application state.
    let state = AppState {
        api_key: args.api_key,
        limit: args.image_limit,
        fallback_solver: match (args.fallback_solver, args.fallback_key) {
            (Some(solver), Some(key)) => Some(
                Solver::builder()
                    .typed(SolverType::from_str(&solver)?)
                    .client(reqwest::Client::new())
                    .client_key(key)
                    .limit(args.image_limit)
                    .endpoint(args.fallback_endpoint)
                    .build(),
            ),
            _ => None,
        },
        onnx: ONNXConfig::builder()
            .model_dir(args.model_dir)
            .update_check(args.update_check)
            .num_threads(args.num_threads)
            .allocator(args.allocator)
            .build(),
    };

    // Create the router.
    let route = Router::new()
        .route("/task", post(task))
        .fallback(handler_404)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::default().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO))
                .on_failure(DefaultOnFailure::new().level(Level::WARN)),
        )
        .with_state(Arc::new(state));

    tracing::info!("Listening on {}", args.bind);

    // Signal the server to shut down using Handle.
    let handle = Handle::new();

    // Spawn a task to gracefully shutdown server.
    tokio::spawn(signal::graceful_shutdown(handle.clone()));

    // If TLS certificate and key are provided, use them.
    match (args.tls_cert, args.tls_key) {
        (Some(cert), Some(key)) => {
            let config = RustlsConfig::from_pem_file(cert, key).await?;
            axum_server::bind_rustls(args.bind, config)
                .handle(handle)
                .serve(route.into_make_service())
                .await?;
        }
        _ => {
            axum_server::bind(args.bind)
                .handle(handle)
                .serve(route.into_make_service())
                .await?;
        }
    }

    Ok(())
}

/// Handle the task
/// This function is responsible for handling tasks. It takes the application state and a task as input.
/// It first validates the task, then tries to convert the task to a variant.
/// Depending on whether a fallback solver is available, it either uses the solver task or the fallback solver task.
/// The function is asynchronous and returns a Result wrapping a JSON TaskResult.
async fn task(
    State(state): State<Arc<AppState>>, // The application state
    Json(task): Json<Task>,             // The task to be handled
) -> Result<Json<TaskResult>> {
    // Validate the task
    validate_task(&state, &task)?;

    // Try to convert task to variant
    let variant_result = Variant::try_from(&task);

    // Match the fallback solver
    let result = match state.fallback_solver.as_ref() {
        None => {
            // If the variant is valid, use solver task
            variant_result.map(|variant| solver_task(&state.onnx, variant, task))
        }
        Some(solver) => {
            // If the variant is valid, use solver task, else use fallback solver task
            match variant_result {
                Ok(variant) => Ok(solver_task(&state.onnx, variant, task)),
                Err(_) => Ok(fallback_solver_task(solver, task)),
            }
        }
    }?;

    result.await
}

/// Handle the model task
/// This function is responsible for handling tasks using the model. It takes the ONNX configuration, a variant, and a task as input.
/// It first gets the model predictor, then processes the task using the predictor.
/// The function is asynchronous and returns a Result wrapping a JSON TaskResult.
/// If the task is successfully processed, it returns a Result wrapping a JSON TaskResult with solved set to true and objects set to the answers.
/// If there is an error during the process, it returns a Result wrapping an error.
async fn solver_task(
    config: &ONNXConfig,
    variant: Variant,
    task: Task,
) -> Result<Json<TaskResult>> {
    // Get the model predictor
    let predictor = onnx::get_predictor(variant, config).await?;

    // Process the task
    let answers = process_image_tasks(task.images, predictor)?;

    // If the task is successfully processed, return the answers
    Ok(Json(
        TaskResult::builder().solved(true).objects(answers).build(),
    ))
}

/// Handle the fallback solver task
/// This function is responsible for handling tasks using the fallback solver. It takes a solver and a task as input.
/// It processes the task using the solver.
/// The function is asynchronous and returns a Result wrapping a JSON TaskResult.
/// If the task is successfully processed, it returns a Result wrapping a JSON TaskResult with solved set to true and objects set to the answers.
/// If there is an error during the process, it returns a Result wrapping an error.
async fn fallback_solver_task(solver: &Solver, task: Task) -> Result<Json<TaskResult>> {
    // Process the task
    let answers = solver.process(task).await?;

    // If the task is successfully processed, return the answers
    Ok(Json(
        TaskResult::builder().solved(true).objects(answers).build(),
    ))
}

/// Process image tasks
/// This function is responsible for processing tasks with one or more images. It takes a vector of images and a predictor as input.
/// It decodes each image, uses the predictor to predict the answer for each image, and collects the answers in a vector.
/// The function returns a Result wrapping a vector of integers.
fn process_image_tasks<P: Predictor + ?Sized>(
    images: Vec<String>, // The images to be processed
    predictor: &P,       // The predictor to be used
) -> Result<Vec<i32>> {
    let mut objects = images
        .into_par_iter()
        .enumerate()
        .map(|(index, image)| {
            let image = decode_image(&image)?; // Decode the image
            let answer = predictor.predict(image)?; // Predict the answer
            Ok((index, answer)) // Return the answer
        })
        .collect::<Result<Vec<(usize, i32)>>>()?;

    objects.sort_by_key(|&(index, _)| index);
    Ok(objects.into_iter().map(|(_, answer)| answer).collect())
}

/// Validate the task
/// This function checks if the task is valid. It takes the application state and a task as input.
/// It first checks if the API key is provided and matches the one in the application state.
/// Then it checks if the number of images in the task is within the limit.
/// If any of these checks fail, it returns an error.
/// If all checks pass, it returns Ok.
fn validate_task(state: &Arc<AppState>, task: &Task) -> Result<()> {
    // Check if API key is provided and matches the one in the state
    match &state.api_key {
        Some(api_key) if task.api_key.as_deref() != Some(api_key) => {
            return Err(Error::InvalidApiKey)
        }
        _ => (),
    }

    // Check if images is empty
    if task.images.is_empty() {
        return Err(Error::InvalidImages);
    }

    // Check if images is greater than limit
    if task.images.len() > state.limit {
        return Err(Error::InvalidSubmitLimit);
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

/// Handles the 404 requests.
async fn handler_404() -> impl IntoResponse {
    Html(
        r#"<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>NOTHING TO SEE HERE</title><style>body{font-family:Arial,sans-serif;background-color:#f8f8f8;margin:0;padding:0;display:flex;justify-content:center;align-items:center;height:100vh;color:#333}.container{text-align:center;max-width:600px;padding:20px;background-color:#fff;box-shadow:0 4px 8px rgba(0,0,0,0.1)}h1{font-size:72px;margin:0;color:#ff6f61}a{display:block;margin-top:20px;color:#ff6f61;text-decoration:none;font-weight:bold;font-size:18px}a:hover{text-decoration:underline}</style></head><body><div class="container"><h1>NOTHING TO SEE HERE</h1></div></body></html>"#,
    )
}
