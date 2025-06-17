use std::{
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

// Type alias to simplify complex types
type EngineReceiver<M> = Arc<
    Mutex<
        mpsc::Receiver<
            InfernumEngineResponse<
                <<M as InfernumModel>::Request as RequestMetadata>::Metadata,
                <M as InfernumModel>::Response,
            >,
        >,
    >,
>;

/// Trait for implementing inference models that can be used with the InfernumEngine.
///
/// Users implement this trait to define their custom model behavior, including
/// the request and response types and the inference logic.
pub trait InfernumModel {
    /// The request type that the model accepts for inference.
    type Request;
    /// The response type that the model returns after inference.
    type Response;
    /// The error type that can be returned during inference.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Runs inference on the given request and returns a response or error.
    fn run(&mut self, request: Self::Request) -> Result<Self::Response, Self::Error>;
}

/// Represents the current state of the inference engine.
#[derive(Clone, Debug, PartialEq)]
pub enum InfernumEngineState {
    /// The engine is idle and ready to accept new inference requests.
    Idle,
    /// The engine is currently processing an inference request.
    Processing,
}

impl InfernumEngineState {
    /// Returns the state as a string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            InfernumEngineState::Idle => "idle",
            InfernumEngineState::Processing => "processing",
        }
    }
}

/// Trait for extracting lightweight metadata from inference requests.
///
/// This allows the engine to store essential information (like prompts) without
/// cloning heavy data (like images) for telemetry and debugging purposes.
pub trait RequestMetadata {
    /// The lightweight metadata type that represents the request.
    type Metadata: Send + 'static;

    /// Extracts lightweight metadata from the request.
    /// This should avoid cloning heavy data like images.
    fn metadata(&self) -> Self::Metadata;
}

/// Internal request wrapper used by the engine to track inference requests.
pub struct InfernumEngineRequest<Req> {
    /// Unique identifier for this inference request.
    pub id: u8,
    /// The actual request data to be processed by the model.
    pub request: Req,
}

/// Response returned by the engine containing both the model's response and telemetry data.
pub struct InfernumEngineResponse<Metadata, Res> {
    /// Unique identifier matching the original request.
    pub id: u8,
    /// Timestamp when the inference started.
    pub start_time: Instant,
    /// Total time taken for the inference.
    pub duration: Duration,
    /// Lightweight metadata extracted from the original request.
    pub request_metadata: Metadata,
    /// The actual response from the model.
    pub response: Res,
}

/// Result type returned when polling for inference results.
pub enum InfernumEngineResult<M: InfernumModel + Send + 'static>
where
    M::Request: RequestMetadata,
{
    /// Successful inference with the response data.
    Success(InfernumEngineResponse<<M::Request as RequestMetadata>::Metadata, M::Response>),
    /// No result available yet, with current engine state.
    Empty(InfernumEngineState),
    /// An error occurred during inference or engine operation.
    Error(String),
}

/// High-performance inference engine that manages model execution in a separate thread.
///
/// The engine provides asynchronous inference capabilities with built-in telemetry,
/// request tracking, and state management. It decouples the API from the model
/// implementation, allowing for flexible request/response types while providing
/// production-ready monitoring capabilities.
pub struct InfernumEngine<M: InfernumModel + Send + 'static>
where
    M::Error: Send + 'static,
    M::Request: Send + RequestMetadata + 'static,
    M::Response: Send + 'static,
{
    state: Arc<Mutex<InfernumEngineState>>,
    req_tx: Option<mpsc::Sender<InfernumEngineRequest<M::Request>>>,
    rep_rx: EngineReceiver<M>,
    inference_handle: Option<JoinHandle<Result<(), M::Error>>>,
    id_counter: Arc<Mutex<u8>>,
}

impl<M: InfernumModel + Send + 'static> InfernumEngine<M>
where
    M::Error: Send + 'static,
    M::Request: Send + RequestMetadata + 'static,
    M::Response: Send + 'static,
{
    /// Creates a new inference engine with the given model.
    ///
    /// The engine will spawn a background thread to handle inference requests
    /// asynchronously. The model will be moved to this background thread.
    ///
    /// # Arguments
    /// * `model` - The model implementation that will handle inference requests
    ///
    /// # Returns
    /// A new `InfernumEngine` instance ready to accept inference requests
    pub fn new(mut model: M) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<InfernumEngineRequest<M::Request>>();
        let (rep_tx, rep_rx) = mpsc::channel::<
            InfernumEngineResponse<<M::Request as RequestMetadata>::Metadata, M::Response>,
        >();
        let state = Arc::new(Mutex::new(InfernumEngineState::Idle));

        let inference_handle = std::thread::spawn({
            let state = state.clone();
            move || -> Result<(), M::Error> {
                while let Ok(req) = req_rx.recv() {
                    log::debug!("Scheduling a new inference");

                    // Extract lightweight metadata before consuming the request
                    let request_metadata = req.request.metadata();

                    *state.lock().unwrap() = InfernumEngineState::Processing;
                    let start_time = Instant::now();

                    let response = model.run(req.request)?;

                    log::debug!("Inference completed");

                    let _ = rep_tx.send(InfernumEngineResponse {
                        id: req.id,
                        start_time,
                        duration: start_time.elapsed(),
                        request_metadata,
                        response,
                    });

                    *state.lock().unwrap() = InfernumEngineState::Idle;
                }
                Ok(())
            }
        });

        Self {
            state,
            req_tx: Some(req_tx),
            rep_rx: Arc::new(Mutex::new(rep_rx)),
            inference_handle: Some(inference_handle),
            id_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Returns the current state of the inference engine.
    pub fn state(&self) -> InfernumEngineState {
        self.state.lock().unwrap().clone()
    }

    /// Attempts to retrieve a completed inference result without blocking.
    ///
    /// # Returns
    /// * `Success` - Contains the inference response with telemetry data
    /// * `Empty` - No result available yet, includes current engine state
    /// * `Error` - An error occurred during inference or engine operation
    pub fn try_poll_response(&self) -> InfernumEngineResult<M> {
        match self.rep_rx.lock().unwrap().try_recv() {
            Ok(response) => InfernumEngineResult::Success(response),
            Err(mpsc::TryRecvError::Empty) => InfernumEngineResult::Empty(self.state()),
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Response channel disconnected");
                InfernumEngineResult::Error("Response channel disconnected".to_string())
            }
        }
    }

    /// Schedules an inference request for asynchronous processing.
    ///
    /// The request will be queued and processed by the background thread.
    /// Each request is assigned a unique ID for tracking purposes.
    ///
    /// # Arguments
    /// * `request` - The inference request to be processed by the model
    pub fn schedule_inference(&self, request: M::Request) {
        if let Some(tx) = &self.req_tx {
            let id = *self.id_counter.lock().unwrap();
            *self.id_counter.lock().unwrap() += 1;
            let _ = tx.send(InfernumEngineRequest { id, request });
        }
    }

    /// Stops the inference engine and shuts down the background thread.
    ///
    /// This method will close the request channel and wait for the background
    /// thread to finish processing any remaining requests.
    pub fn stop(&mut self) {
        self.req_tx.take();
        if let Some(handle) = self.inference_handle.take() {
            let _ = handle.join();
        }
    }
}

impl<M: InfernumModel + Send + 'static> Drop for InfernumEngine<M>
where
    M::Error: Send + 'static,
    M::Request: Send + RequestMetadata + 'static,
    M::Response: Send + 'static,
{
    fn drop(&mut self) {
        self.stop();
    }
}
