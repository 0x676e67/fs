mod signal;
mod solver;
mod task;

use std::str::FromStr;
use std::sync::Arc;

use self::solver::Solver;
use self::task::Task;
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
    solver: Option<Solver>,
    // ONNX configuration
    onnx_config: onnx::ONNXConfig,
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
        solver: match (args.fallback_solver, args.fallback_key) {
            (Some(solver), Some(key)) => Some(Solver::new(
                SolverType::from_str(&solver)?,
                key,
                args.fallback_endpoint,
                args.fallback_image_limit,
            )),
            _ => None,
        },
        onnx_config: onnx::ONNXConfig {
            model_dir: args.model_dir,
            update_check: args.update_check,
            num_threads: args.num_threads,
            allocator: args.allocator,
        },
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
async fn task(
    State(state): State<Arc<AppState>>,
    Json(task): Json<Task>,
) -> Result<Json<TaskResult>> {
    // Validate the task
    validate_task(&state, &task)?;

    // If the variant is valid, use fallback solver
    if let Ok(model) = Variant::from_str(task.game_variant_instructions.0.as_str()) {
        // handle the solver task
        handle_solver_task(&state.onnx_config, task, model).await
    } else {
        // handle the fallback solver task
        handle_fallback_solver_task(&state, task).await
    }
}

/// Handle the model task
async fn handle_solver_task(
    args: &ONNXConfig,
    task: Task,
    model: Variant,
) -> Result<Json<TaskResult>> {
    // Get the model predictor
    let predictor = onnx::get_predictor(model, args).await?;

    // Handle the single or multiple image task
    let answers = if task.images.len() == 1 {
        handle_single_image_task(&task.images[0], predictor)?
    } else {
        handle_multiple_images_task(task.images, predictor)?
    };

    let result = TaskResult {
        error: None,
        solved: true,
        objects: Some(answers),
    };
    Ok(Json(result))
}

/// Handle the single image task
fn handle_single_image_task<P: Predictor + ?Sized>(
    image: &String,
    predictor: &P,
) -> Result<Vec<i32>> {
    let image = decode_image(image)?;
    let answer = predictor.predict(image)?;
    Ok(vec![answer])
}

/// Handle the multiple images task
fn handle_multiple_images_task<P: Predictor + ?Sized>(
    images: Vec<String>,
    predictor: &P,
) -> Result<Vec<i32>> {
    let mut objects = images
        .into_par_iter()
        .enumerate()
        .map(|(index, image)| {
            let image = decode_image(&image)?;
            let answer = predictor.predict(image)?;
            Ok((index, answer))
        })
        .collect::<Result<Vec<(usize, i32)>>>()?;

    objects.sort_by_key(|&(index, _)| index);
    Ok(objects.into_iter().map(|(_, answer)| answer).collect())
}

/// Handle the fallback solver task
async fn handle_fallback_solver_task(
    state: &Arc<AppState>,
    task: Task,
) -> Result<Json<TaskResult>> {
    // Get the fallback solver
    let solver = state.solver.as_ref().unwrap();
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
                    .await?;
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
                    .await?;
                answers.extend(answer);
            }
        }
    }

    let result = TaskResult {
        error: None,
        solved: true,
        objects: Some(answers),
    };

    Ok(Json(result))
}

/// Validate task
fn validate_task(state: &Arc<AppState>, task: &Task) -> Result<()> {
    // If API key is not provided, return error
    state.api_key.as_deref().map_or(Ok(()), |api_key| {
        if task.api_key.as_deref() == Some(api_key) {
            Ok(())
        } else {
            Err(Error::InvalidApiKey)
        }
    })?;

    // If images is empty, return error
    if task.images.is_empty() {
        return Err(Error::InvalidImages);
    }

    // If images is greater than limit, return error
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
