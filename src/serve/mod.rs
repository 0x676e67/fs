mod signal;
mod solver;
mod task;

use self::solver::FallbackSolver;
pub use self::task::Task;
use crate::{
    error::Error,
    onnx::{ONNXConfig, ONNXFetch},
    serve::solver::TypedFallback,
    BootArgs, Result,
};
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::post,
    Json, Router,
};
use axum_server::{tls_rustls::RustlsConfig, Handle};
use solver::{ONNXSolver, Solver, SolverHelper};
use std::{str::FromStr, sync::Arc};
pub use task::TaskResult;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnResponse, TraceLayer};
use tracing::Level;

/// Application state
struct AppState {
    // API key
    api_key: Option<String>,
    // Solver
    solver: SolverHelper,
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

    // Print boot arguments.
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Limit: {}", args.limit);
    tracing::info!("Model dir: {:?}", args.model_dir);
    tracing::info!("Update check: {}", args.update_check);
    tracing::info!("Threads: {}", args.num_threads);
    tracing::info!("Allocator: {:?}", args.allocator);

    // Initialize the application state.
    let state = AppState {
        api_key: args.api_key,
        solver: SolverHelper::builder()
            .limit(args.limit)
            .onnx_solver(
                ONNXSolver::builder()
                    .onnx(
                        ONNXConfig::builder()
                            .model_dir(args.model_dir)
                            .update_check(args.update_check)
                            .num_threads(args.num_threads)
                            .allocator(args.allocator)
                            .onnx_store(ONNXFetch::new(args.store).await)
                            .build(),
                    )
                    .predictors(Default::default())
                    .build(),
            )
            .fallback_solver(match (args.fallback_solver, args.fallback_key) {
                (Some(solver), Some(key)) => Some(
                    FallbackSolver::builder()
                        .typed(TypedFallback::from_str(&solver)?)
                        .client(reqwest::Client::new())
                        .client_key(key)
                        .endpoint(args.fallback_endpoint)
                        .limit(args.fallback_image_limit)
                        .build(),
                ),
                _ => None,
            })
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
/// This function is responsible for handling tasks. It takes the application
/// state and a task as input. It first validates the task, then tries to
/// convert the task to a variant. Depending on whether a fallback solver is
/// available, it either uses the solver task or the fallback solver task.
/// The function is asynchronous and returns a Result wrapping a JSON
/// TaskResult.
async fn task(
    State(state): State<Arc<AppState>>,
    Json(task): Json<Task>,
) -> Result<Json<TaskResult>> {
    // Check if API key is provided and matches the one in the state
    match &state.api_key {
        Some(api_key) if task.api_key.as_deref() != Some(api_key) => {
            return Err(Error::InvalidApiKey);
        }
        _ => (),
    }

    // Process the solver task
    state.solver.process(Arc::new(task)).await
}

/// Handles the 404 requests.
async fn handler_404() -> impl IntoResponse {
    Html(
        r#"<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>NOTHING TO SEE HERE</title><style>body{font-family:Arial,sans-serif;background-color:#f8f8f8;margin:0;padding:0;display:flex;justify-content:center;align-items:center;height:100vh;color:#333}.container{text-align:center;max-width:600px;padding:20px;background-color:#fff;box-shadow:0 4px 8px rgba(0,0,0,0.1)}h1{font-size:72px;margin:0;color:#ff6f61}a{display:block;margin-top:20px;color:#ff6f61;text-decoration:none;font-weight:bold;font-size:18px}a:hover{text-decoration:underline}</style></head><body><div class="container"><h1>NOTHING TO SEE HERE</h1></div></body></html>"#,
    )
}
