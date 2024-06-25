use super::{Task, TaskResult};
use crate::{
    error::Error,
    onnx::{self, ONNXConfig, Predictor, Variant},
    Result,
};
use axum::Json;
use image::DynamicImage;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{str::FromStr, sync::Arc};
use tokio::sync::OnceCell;
use typed_builder::TypedBuilder;

/// The `SolverProcess` trait defines a common interface for processing tasks.
///
/// This trait is intended to be implemented by different types of solvers,
/// each of which may process tasks in a different way.
///
/// The `process` method takes a `Task` as input and returns a `Result`
/// containing a `Json<TaskResult>`. This allows for flexibility in the
/// processing logic and the format of the results.
pub trait Solver {
    /// Process a given task.
    ///
    /// # Parameters
    /// - `task`: The task to be processed.
    ///
    /// # Returns
    /// - A `Result` containing a `Json<TaskResult>` if the task is processed successfully.
    /// - An `Error` if the task processing fails.
    async fn process(&self, task: Arc<Task>) -> Result<Json<TaskResult>>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedFallback {
    Yescaptcha,
    Capsolver,
}

impl FromStr for TypedFallback {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "yescaptcha" => Ok(Self::Yescaptcha),
            "capsolver" => Ok(Self::Capsolver),
            _ => Err(Error::InvalidSolverType(s.to_string())),
        }
    }
}

/// `SolverHelper` is a struct that encapsulates the logic for handling tasks using both an ONNX model and a fallback solver.
///
/// It contains an `ONNXSolver` instance, which handles tasks using an ONNX model,
/// an optional `FallbackSolver` instance, which handles tasks using a fallback solver if the ONNX model fails,
/// and a limit for the number of tasks that can be processed.
///
/// # Fields
/// * `limit`: The maximum number of tasks that can be processed.
/// * `onnx_solver`: The ONNX solver used to process tasks.
/// * `fallback_solver`: The fallback solver used to process tasks if the ONNX solver fails.
#[derive(TypedBuilder)]
pub struct SolverHelper {
    limit: usize,
    onnx_solver: DefaultSolver,
    fallback_solver: Option<FallbackSolver>,
}

impl SolverHelper {
    /// Validate the task
    /// This function checks if the task is valid. It takes the application state
    /// and a task as input. It first checks if the API key is provided and matches
    /// the one in the application state. Then it checks if the number of images in
    /// the task is within the limit. If any of these checks fail, it returns an
    /// error. If all checks pass, it returns Ok.
    fn validate_task(&self, task: &Task) -> Result<()> {
        // Check if images is empty
        if task.images.is_empty() {
            return Err(Error::InvalidImages);
        }

        // Check if images is greater than limit
        if task.images.len() > self.limit {
            return Err(Error::InvalidSubmitLimit);
        }

        Ok(())
    }
}

impl Solver for SolverHelper {
    /// Process the task
    async fn process(&self, task: Arc<Task>) -> Result<Json<TaskResult>> {
        // Validate the task
        self.validate_task(&task)?;

        // Match the fallback solver
        match self.fallback_solver.as_ref() {
            // If there is no fallback solver, use the solver task
            None => self.onnx_solver.process(task).await,
            // If there is a fallback solver, use the fallback solver task
            Some(fallback_solver) => {
                // Try to use the solver task
                match self.onnx_solver.process(task.clone()).await {
                    // If the solver task is successful, return the result
                    Ok(result) => Ok(result),
                    // If the solver task fails, use the fallback solver task
                    Err(_) => fallback_solver.process(task).await,
                }
            }
        }
    }
}

/// `OnnxSolver` is a struct that encapsulates the logic for handling tasks using an ONNX model.
///
/// It contains an `ONNXConfig` instance, which holds the configuration for the ONNX model,
/// and an array of `OnceCell` instances, each of which can hold a `Box<dyn Predictor>`.
/// The `OnceCell` instances are used to lazily initialize and store the predictors for each variant.
///
/// # Fields
/// * `onnx`: The ONNX model configuration.
/// * `predictors`: An array of `OnceCell` instances, each of which can hold a `Box<dyn Predictor>`.
#[derive(TypedBuilder)]
pub struct DefaultSolver {
    onnx: ONNXConfig,
    predictors: [OnceCell<Box<dyn Predictor>>; Variant::const_count()],
}

impl Solver for DefaultSolver {
    async fn process(&self, task: Arc<Task>) -> Result<Json<TaskResult>> {
        // Try to convert the task to a variant
        let variant = Variant::try_from(&*task)?;

        // Process the task using the model
        let predictor = self.predictors[variant as usize]
            .get_or_try_init(|| onnx::new_predictor(variant, &self.onnx))
            .await?;

        // Check if the predictor is active
        if !predictor.active() {
            return Err(Error::PredictorNotActive(variant));
        }

        // Process the task
        let answers = {
            let mut objects = task
                .images
                .par_iter()
                .enumerate()
                .map(|(index, image)| {
                    let image = decode_image(image)?;
                    let answer = predictor.predict(image)?;
                    Ok((index, answer))
                })
                .collect::<Result<Vec<(usize, i32)>>>()?;

            objects.sort_by_key(|&(index, _)| index);
            objects.into_iter().map(|(_, answer)| answer).collect()
        };

        // If the task is successfully processed, return the answers
        Ok(Json(
            TaskResult::builder().solved(true).objects(answers).build(),
        ))
    }
}

/// `FallbackSolver` is a struct that encapsulates the logic for handling tasks using a fallback solver.
///
/// It contains a `TypedFallback` instance, which holds the configuration for the fallback solver,
/// a `reqwest::Client` instance for making HTTP requests, a client key for authentication,
/// an optional endpoint URL, and a limit for the number of tasks that can be processed.
///
/// # Fields
/// * `typed`: The fallback solver configuration.
/// * `client`: The HTTP client used to make requests to the fallback solver.
/// * `client_key`: The client key used for authentication with the fallback solver.
/// * `endpoint`: The endpoint URL of the fallback solver. If not provided, a default endpoint is used based on the `SolverType`.
/// * `limit`: The maximum number of tasks that can be processed by the fallback solver.
#[derive(TypedBuilder)]
pub struct FallbackSolver {
    typed: TypedFallback,
    client: reqwest::Client,
    client_key: String,
    endpoint: Option<String>,
    limit: usize,
}

impl FallbackSolver {
    /// This method is responsible for submitting a task to the solver.
    ///
    /// It takes a `SubmitTask` object as an argument, which contains the
    /// details of the task to be submitted. Depending on the `SolverType`,
    /// it prepares the endpoint URL and the request body. For `SolverType::Yescaptcha`, it uses the endpoint "https://api.yescaptcha.com/createTask" by default.
    /// For `SolverType::Capsolver`, it uses the endpoint "https://api.capsolver.com/createTask" by default.
    /// The request body is a JSON object that includes the client key, task
    /// details (type, image(s), and question), and either a softID or an appId.
    /// After preparing the endpoint and the request body, it sends a request to
    /// the solver.
    ///
    /// This method is asynchronous and returns a `Result<Vec<i32>>`.
    /// If the task is successfully submitted and solved, it returns a `Result`
    /// wrapping a vector of integers. If there is an error during the
    /// process, it returns a `Result` wrapping an error.
    async fn submit_task(&self, submit_task: SubmitTask<'_>) -> Result<Vec<i32>> {
        let (endpoint, body) = match self.typed {
            TypedFallback::Yescaptcha => (
                self.endpoint
                    .as_deref()
                    .unwrap_or("https://api.yescaptcha.com/createTask"),
                json!({
                    "clientKey": self.client_key,
                    "task": {
                        "type": "FunCaptchaClassification",
                        "image": submit_task.image,
                        "question": &submit_task.game_variant_instructions.1,
                    },
                    "softID": "26299"
                }),
            ),
            TypedFallback::Capsolver => (
                self.endpoint
                    .as_deref()
                    .unwrap_or("https://api.capsolver.com/createTask"),
                json!({
                    "clientKey": self.client_key,
                    "task": {
                        "type": "FunCaptchaClassification",
                        "images": submit_task.images,
                        "question": submit_task.game_variant_instructions.0
                    },
                    "appId": "60632CB0-8BE8-41D3-808F-60CC2442F16E"
                }),
            ),
        };

        // Send request
        let resp = self
            .client
            .post(endpoint)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        // Task response
        let task = resp.json::<TaskResp0>().await?;
        // If error
        if let Some(error_description) = task.error_description {
            return Err(Error::FallbackSolverError(error_description));
        }

        Ok(task.solution.objects)
    }
}

impl Solver for FallbackSolver {
    async fn process(&self, task: Arc<Task>) -> Result<Json<TaskResult>> {
        // Get game variant and instructions
        let (game_variant, instructions) = &task.game_variant_instructions;

        // Answers
        let mut answers = Vec::with_capacity(task.images.len());

        match self.typed {
            TypedFallback::Yescaptcha => {
                // single image
                for image in task.images.iter() {
                    // submit task
                    let task = SubmitTask::builder()
                        .image(image)
                        .game_variant_instructions((&game_variant, &instructions))
                        .build();
                    let answer = self.submit_task(task).await?;
                    answers.extend(answer);
                }
            }
            TypedFallback::Capsolver => {
                // split chunk images
                let images_chunk = task.images.chunks(self.limit.max(1));

                // submit multiple images task
                for chunk in images_chunk {
                    // submit task
                    let task = SubmitTask::builder()
                        .images(chunk)
                        .game_variant_instructions((&game_variant, &instructions))
                        .build();
                    let answer = self.submit_task(task).await?;
                    answers.extend(answer);
                }
            }
        }

        Ok(Json(
            TaskResult::builder().solved(true).objects(answers).build(),
        ))
    }
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

#[derive(Deserialize, Default)]
#[serde(default)]
struct TaskResp0 {
    #[serde(rename = "errorId")]
    error_id: i32,
    #[serde(rename = "errorCode")]
    error_code: String,
    #[serde(rename = "errorDescription")]
    error_description: Option<String>,
    status: String,
    solution: SolutionResp,
    #[serde(rename = "taskId")]
    task_id: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SolutionResp {
    objects: Vec<i32>,
}

#[derive(typed_builder::TypedBuilder)]
pub struct SubmitTask<'a> {
    #[builder(default, setter(strip_option))]
    pub image: Option<&'a String>,
    #[builder(default, setter(strip_option))]
    pub images: Option<&'a [String]>,
    pub game_variant_instructions: (&'a str, &'a str),
}
