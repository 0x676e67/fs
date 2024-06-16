use std::str::FromStr;

use crate::error::Error;
use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

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

#[derive(Clone)]
pub struct Solver {
    typed: SolverType,
    client: reqwest::Client,
    client_key: String,
    limit: usize,
    endpoint: Option<String>,
}

impl Solver {
    pub fn new(
        typed: SolverType,
        client_key: String,
        endpoint: Option<String>,
        limit: usize,
    ) -> Self {
        Self {
            typed,
            client_key,
            endpoint,
            limit,
            client: reqwest::Client::new(),
        }
    }

    /// Get the solver image limit
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Get the solver type
    pub fn solver(&self) -> &SolverType {
        &self.typed
    }

    pub async fn submit_task(&self, submit_task: SubmitTask<'_>) -> Result<Vec<i32>> {
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

pub struct SubmitTask<'a> {
    pub image: Option<&'a String>,
    pub images: Option<&'a [String]>,
    pub game_variant_instructions: (&'a str, &'a str),
}
