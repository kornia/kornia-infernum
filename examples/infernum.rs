use argh::FromArgs;
use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
};
use kornia_infernum::{
    engine::{InfernumEngine, InfernumEngineRequest, InfernumEngineResult, InfernumEngineState},
    model::{InfernumModel, InfernumModelRequest, InfernumModelResponse},
};
use kornia_paligemma::{Paligemma, PaligemmaConfig, PaligemmaError};
use reqwest::StatusCode;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

mod messages;

// defaults for the server
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: u16 = 3000;

#[derive(FromArgs)]
/// Infernum is a tool for running inference on images.
struct InfernumArgs {
    /// the host to run the server on
    #[argh(option, short = 'h', default = "DEFAULT_HOST.to_string()")]
    host: String,

    /// the port to run the server on
    #[argh(option, short = 'p', default = "DEFAULT_PORT")]
    port: u16,
}

async fn post_inference(
    State(engine): State<Arc<InfernumEngine<PaligemmaModel>>>,
    Json(payload): Json<messages::InferenceRequest>,
) -> impl IntoResponse {
    if engine.state() != InfernumEngineState::Idle {
        log::debug!("Engine is still processing");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Engine is still processing" })),
        );
    }

    // Read image based on extension
    let img = match read_image_from_path(&payload.image_path) {
        Ok(img) => img,
        Err(error_msg) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": error_msg })));
        }
    };

    // schedule the inference
    engine.schedule_inference(InfernumEngineRequest {
        id: 0,
        prompt: payload.prompt.clone(),
        image: img,
    });

    log::info!("Scheduled inference successfully");

    (StatusCode::OK, Json(json!({ "status": "scheduled" })))
}

async fn get_result(
    State(engine): State<Arc<InfernumEngine<PaligemmaModel>>>,
) -> impl IntoResponse {
    // If we're here, there should be a result available
    match engine.try_poll_response() {
        InfernumEngineResult::Success(response) => {
            log::info!("Result received successfully");
            let inference_response = messages::InferenceResponse {
                prompt: response.prompt,
                start_time: response.start_time.elapsed().as_nanos(),
                duration: response.duration,
                response: response.response,
            };

            (
                StatusCode::OK,
                Json(json!({
                    "status": "success",
                    "response": inference_response
                })),
            )
        }
        InfernumEngineResult::Empty(state) => {
            log::warn!("Expected a result but none was available");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    json!({ "status": state.as_str(), "message": "Expected result not available" }),
                ),
            )
        }
        InfernumEngineResult::Error(e) => {
            // This is an unexpected state - we should have a result
            log::warn!("Expected a result but none was available");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": "error", "message": e })),
            )
        }
    }
}

// Helper function
fn read_image_from_path(path: &PathBuf) -> Result<kornia_image::Image<u8, 3>, String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| "Invalid file extension".to_string())?;

    match extension {
        "jpg" | "jpeg" => kornia_io::jpeg::read_image_jpeg_rgb8(path).map_err(|e| e.to_string()),
        "png" => kornia_io::png::read_image_png_rgb8(path).map_err(|e| e.to_string()),
        _ => Err(format!("Unsupported image format: {}", extension)),
    }
}

// custom model that uses Paligemma to run inference
struct PaligemmaModel(Paligemma);

impl InfernumModel for PaligemmaModel {
    type Error = PaligemmaError;

    fn run(&mut self, request: InfernumModelRequest) -> Result<InfernumModelResponse, Self::Error> {
        let response = self
            .0
            .inference(&request.image, &request.prompt, 50, false)?;

        Ok(InfernumModelResponse { response })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: InfernumArgs = argh::from_env();

    // format the host and port
    let addr = format!("{}:{}", args.host, args.port);

    let model = Paligemma::new(PaligemmaConfig::default())?;
    let engine = Arc::new(InfernumEngine::new(PaligemmaModel(model)));

    let app = Router::new()
        .route("/", get(|| async { "Welcome to Infernum!" }))
        .route("/inference", post(post_inference))
        .route("/results", get(get_result))
        .with_state(engine);

    log::info!("🚀 Starting the server");
    log::info!("🔥 Listening on: {}", addr);
    log::info!("🔧 Press Ctrl+C to stop the server");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
