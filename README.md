# 🔥 Kornia Infernum

A Rust library for running inference on Visual Language Models. Kornia Infernum provides a simple API for running image-based inference via a clean and efficient threaded API to schedule and poll for results.

## ✨ Features

- 🚀 Simple API for Visual Language Model Inference
- ⚡ Asynchronous processing with thread-based inference engine
- 🧠 Built-in state management for inference requests
- 🌐 Example server implementation with REST API

## 📦 Installation

Add Kornia Infernum to your `Cargo.toml`:

```toml
[dependencies]
kornia-infernum = "0.1.0"
```

## 🚀 Quick Start

```rust
use kornia_image::Image;
use kornia_infernum::{
    InfernumEngine, InfernumEngineRequest, InfernumEngineResult,
    model::{InfernumModel, InfernumModelRequest, InfernumModelResponse}
};
use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_paligemma::{Paligemma, PaligemmaConfig, PaligemmaError};

// Create a wrapper for PaliGemma
pub struct PaligemmaModel(Paligemma);

impl InfernumModel for PaligemmaModel {
    type Error = PaligemmaError;

    fn run(&mut self, request: InfernumModelRequest) -> Result<InfernumModelResponse, Self::Error> {
        let response = self.0.inference(&request.image, &request.prompt, 50, false)?;
        Ok(InfernumModelResponse { response })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the model
    let paligemma = Paligemma::new(PaligemmaConfig::default())?;
    let model = PaligemmaModel(paligemma);
    
    // Initialize the engine with the model
    let engine = InfernumEngine::new(model);
    
    // Load an image
    let image = read_image_jpeg_rgb8("path/to/image.jpg")?;
    
    // Create a request
    let request = InfernumEngineRequest {
        id: 0,
        prompt: "Describe this image".to_string(),
        image,
    };
    
    // Schedule inference
    engine.schedule_inference(request);
    
    // Poll for results
    loop {
        match engine.try_poll_response() {
            InfernumEngineResult::Success(response) => {
                println!("Response: {}", response.response);
                break;
            }
            InfernumEngineResult::Empty(state) => {
                println!("Engine state: {}", state.as_str());
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            InfernumEngineResult::Error(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    
    Ok(())
}
```

## 🌐 Example Server

Kornia Infernum includes a ready-to-use HTTP server based on Axum. Run it with:

```bash
cargo run --example infernum
```

The server provides endpoints for submitting inference requests and retrieving results:

- 📤 `POST /inference` - Submit an image with a prompt for inference
- 📥 `GET /results` - Retrieve inference results

Use the included client to interact with the server:

```bash
# Run inference
cargo run --example client -- inference -i path/to/image.jpg -p "Describe this image"

# Check results
cargo run --example client -- results
```

## 🔧 Requirements

- 🦀 Rust 2024 edition
- Additional system dependencies may be required for CUDA support

## 📜 License

Licensed under the Apache License, Version 2.0.

## 👏 Acknowledgments

- Part of the [Kornia](https://github.com/kornia) ecosystem for computer vision in Rust