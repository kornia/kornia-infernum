use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub image_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceResponse {
    pub prompt: String,
    pub start_time: u128,
    pub duration: Duration,
    pub response: String,
}
