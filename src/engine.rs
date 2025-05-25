use crate::model::{InfernumModel, InfernumModelRequest};
use kornia_image::Image;
use std::{
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, PartialEq)]
pub enum InfernumEngineState {
    Idle,
    Processing,
}

impl InfernumEngineState {
    pub fn as_str(&self) -> &'static str {
        match self {
            InfernumEngineState::Idle => "idle",
            InfernumEngineState::Processing => "processing",
        }
    }
}

pub struct InfernumEngineRequest {
    pub id: u8,
    pub prompt: String,
    pub image: Image<u8, 3>,
}

pub struct InfernumEngineResponse {
    pub id: u8,
    pub prompt: String,
    pub start_time: Instant,
    pub duration: Duration,
    pub response: String,
}

pub enum InfernumEngineResult {
    Success(InfernumEngineResponse),
    Empty(InfernumEngineState),
    Error(String),
}

pub struct InfernumEngine<M: InfernumModel + Send + 'static> {
    state: Arc<Mutex<InfernumEngineState>>,
    req_tx: Option<mpsc::Sender<InfernumEngineRequest>>,
    rep_rx: Arc<Mutex<mpsc::Receiver<InfernumEngineResponse>>>,
    inference_handle: Option<JoinHandle<Result<(), M::Error>>>,
}

impl<M: InfernumModel + Send + 'static> InfernumEngine<M>
where
    M::Error: Send + 'static,
{
    pub fn new(mut model: M) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<InfernumEngineRequest>();
        let (rep_tx, rep_rx) = mpsc::channel::<InfernumEngineResponse>();
        let state = Arc::new(Mutex::new(InfernumEngineState::Idle));

        let inference_handle = std::thread::spawn({
            let state = state.clone();
            move || -> Result<(), M::Error> {
                while let Ok(req) = req_rx.recv() {
                    log::debug!("Scheduling a new inference");

                    // Extract values before move
                    let (req_id, req_prompt) = (req.id, req.prompt.clone());

                    *state.lock().unwrap() = InfernumEngineState::Processing;
                    let start_time = Instant::now();

                    let response = model.run(InfernumModelRequest {
                        prompt: req_prompt.clone(),
                        image: req.image,
                    })?;

                    log::debug!("Inference completed");

                    let _ = rep_tx.send(InfernumEngineResponse {
                        id: req_id,
                        prompt: req_prompt,
                        start_time,
                        duration: start_time.elapsed(),
                        response: response.response,
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
        }
    }

    pub fn state(&self) -> InfernumEngineState {
        self.state.lock().unwrap().clone()
    }

    pub fn try_poll_response(&self) -> InfernumEngineResult {
        match self.rep_rx.lock().unwrap().try_recv() {
            Ok(response) => InfernumEngineResult::Success(response),
            Err(mpsc::TryRecvError::Empty) => InfernumEngineResult::Empty(self.state()),
            Err(mpsc::TryRecvError::Disconnected) => {
                log::error!("Response channel disconnected");
                InfernumEngineResult::Error("Response channel disconnected".to_string())
            }
        }
    }

    pub fn schedule_inference(&self, req: InfernumEngineRequest) {
        if let Some(tx) = &self.req_tx {
            let _ = tx.send(req);
        }
    }

    pub fn stop(&mut self) {
        self.req_tx.take();
        if let Some(handle) = self.inference_handle.take() {
            let _ = handle.join();
        }
    }
}

impl<M: InfernumModel + Send + 'static> Drop for InfernumEngine<M> {
    fn drop(&mut self) {
        self.stop();
    }
}
