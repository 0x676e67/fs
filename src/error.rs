use axum::{response::IntoResponse, Json};

use crate::serve::TaskResult;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    SelfUpdateError(#[from] self_update::errors::Error),

    #[error("fc is not running")]
    NotRunning,

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("Invalid allocator: {0}")]
    InvalidAllocator(String),

    #[error("Invalid submit limit")]
    InvalidSubmitLimit,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Invalid images")]
    InvalidImages,

    #[error("Invalid input image size: {0:?}")]
    InvalidImageSize((u32, u32)),

    #[error(transparent)]
    ImageError(#[from] image::ImageError),

    #[error("unknown variant type: {0}")]
    UnknownVariantType(String),

    #[error("model name is not valid: {0}")]
    InvalidModelName(String),

    #[error(transparent)]
    ImageDecodeError(#[from] base64::DecodeError),

    #[error("Fallback solver error: {0}")]
    FallbackSolverError(String),

    #[error("Invalid solver type: {0}")]
    InvalidSolverType(String),

    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),

    #[error(transparent)]
    OnnxError(#[from] ort::Error),

    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error),

    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status = match self {
            Error::IoError(_)
            | Error::SerdeJsonError(_)
            | Error::OnnxError(_)
            | Error::InvalidModelName(_)
            | Error::ReqwestError(_)
            | Error::FallbackSolverError(_) => StatusCode::INTERNAL_SERVER_ERROR,

            Error::InvalidSubmitLimit
            | Error::InvalidApiKey
            | Error::InvalidImages
            | Error::ImageDecodeError(_)
            | Error::InvalidAllocator(_)
            | Error::InvalidSolverType(_)
            | Error::UnknownVariantType(_)
            | Error::InvalidImageSize(_)
            | Error::ShapeError(_)
            | Error::ImageError(_) => StatusCode::BAD_REQUEST,

            _ => StatusCode::BAD_GATEWAY,
        };

        (
            status,
            Json(
                TaskResult::builder()
                    .error(self.to_string())
                    .solved(false)
                    .build(),
            ),
        )
            .into_response()
    }
}
