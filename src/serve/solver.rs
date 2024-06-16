use super::Task;
use crate::error::Error;
use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::str::FromStr;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverType {
    Yescaptcha,
    Capsolver,
}

impl FromStr for SolverType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "yescaptcha" => Ok(Self::Yescaptcha),
            "capsolver" => Ok(Self::Capsolver),
            _ => Err(Error::InvalidSolverType(s.to_string())),
        }
    }
}

/// `Solver` is a structure that represents a solver for captcha tasks.
///
/// It has the following fields:
/// - `typed`: A `SolverType` enum that represents the type of the solver. It can be either `Yescaptcha` or `Capsolver`.
/// - `client`: A `reqwest::Client` object that is used to send HTTP requests.
/// - `client_key`: A `String` that represents the client key for the solver API.
/// - `limit`: A `usize` that represents the maximum number of images that can be processed in a single task.
/// - `endpoint`: An `Option<String>` that represents the endpoint URL of the solver API. It is optional and can be `None` if the default endpoint is used.
///
/// The `Solver` structure is created using the `typed_builder::TypedBuilder` derive macro, which provides a builder pattern for creating a `Solver` object.
#[derive(typed_builder::TypedBuilder)]
pub struct Solver {
    typed: SolverType,
    client: reqwest::Client,
    client_key: String,
    limit: usize,
    endpoint: Option<String>,
}

impl Solver {
    /// This method is responsible for processing a task based on the solver type.
    ///
    /// It takes a `Task` object as an argument, which contains the details of the task to be processed.
    /// The task details include the game variant, instructions, and images.
    ///
    /// The method initializes an empty vector `answers` to store the answers for the task images.
    ///
    /// Depending on the `SolverType`, it processes the task differently:
    /// - For `SolverType::Yescaptcha`, it processes each image individually. For each image, it creates a `SubmitTask` object and submits the task using the `submit_task` method. The answer for the task is then added to the `answers` vector.
    /// - For `SolverType::Capsolver`, it splits the task images into chunks with a maximum size defined by `self.limit`. Each chunk of images is then processed as a separate task.
    ///
    /// This method is asynchronous and returns a `Result<Vec<i32>>`.
    /// If the task is successfully processed, it returns a `Result` wrapping a vector of integers representing the answers.
    /// If there is an error during the process, it returns a `Result` wrapping an error.
    pub async fn process(&self, task: Task) -> Result<Vec<i32>> {
        // Get game variant and instructions
        let (game_variant, instructions) = task.game_variant_instructions;

        // Answers
        let mut answers = Vec::with_capacity(task.images.len());

        match self.typed {
            SolverType::Yescaptcha => {
                // single image
                for image in task.images {
                    // submit task
                    let task = SubmitTask::builder()
                        .image(&image)
                        .game_variant_instructions((&game_variant, &instructions))
                        .build();
                    let answer = self.submit_task(task).await?;
                    answers.extend(answer);
                }
            }
            SolverType::Capsolver => {
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

        Ok(answers)
    }

    /// This method is responsible for submitting a task to the solver.
    ///
    /// It takes a `SubmitTask` object as an argument, which contains the details of the task to be submitted.
    /// Depending on the `SolverType`, it prepares the endpoint URL and the request body.
    /// For `SolverType::Yescaptcha`, it uses the endpoint "https://api.yescaptcha.com/createTask" by default.
    /// For `SolverType::Capsolver`, it uses the endpoint "https://api.capsolver.com/createTask" by default.
    /// The request body is a JSON object that includes the client key, task details (type, image(s), and question), and either a softID or an appId.
    /// After preparing the endpoint and the request body, it sends a request to the solver.
    ///
    /// This method is asynchronous and returns a `Result<Vec<i32>>`.
    /// If the task is successfully submitted and solved, it returns a `Result` wrapping a vector of integers.
    /// If there is an error during the process, it returns a `Result` wrapping an error.
    async fn submit_task(&self, submit_task: SubmitTask<'_>) -> Result<Vec<i32>> {
        let (endpoint, body) = match self.typed {
            SolverType::Yescaptcha => (
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
            SolverType::Capsolver => (
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
