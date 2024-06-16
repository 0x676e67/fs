use axum::{response::IntoResponse, Json};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    SelfUpdateError(#[from] self_update::errors::Error),

    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),

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

    #[error(transparent)]
    ImageError(#[from] image::ImageError),

    #[error(transparent)]
    ImageDecodeError(#[from] base64::DecodeError),
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let (status, msg) = match self {
            Error::AnyhowError(_) | Error::SelfUpdateError(_) | Error::IoError(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            Error::InvalidSubmitLimit
            | Error::InvalidApiKey
            | Error::InvalidImages
            | Error::ImageDecodeError(_)
            | Error::InvalidAllocator(_)
            | Error::ImageError(_) => (StatusCode::BAD_REQUEST, self.to_string()),
        };

        (
            status,
            Json(serde_json::json!({
                "error": msg,
            })),
        )
            .into_response()
    }
}
