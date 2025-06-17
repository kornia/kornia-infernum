# �� Kornia Infernum

A high-performance Rust library for running inference on AI models with built-in telemetry and production-ready features. Kornia Infernum provides a flexible, threaded inference engine that decouples model implementation from API design while delivering rich monitoring capabilities.

## ✨ Features

- 🚀 **Flexible Model Integration** - Support for any model through trait-based design
- ⚡ **Asynchronous Processing** - Non-blocking inference with background thread execution
- 📊 **Built-in Telemetry** - Request tracking, timing, and metadata collection
- 🎯 **Production Ready** - State management, error handling, and monitoring
- 🪶 **Lightweight Metadata** - Avoid cloning heavy data while preserving essential information
- 🔧 **Type-Safe API** - Fully generic with compile-time guarantees

## 📦 Installation

Add Kornia Infernum to your `Cargo.toml`:

```toml
[dependencies]
kornia-infernum = "0.1.0"
```

## 🚀 Quick Start

### 1. Implement Your Model

```rust
use kornia_infernum::{InfernumModel, RequestMetadata};
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};

// Define your request and response types
#[derive(Clone)]
struct MyRequest {
    image: Image<u8, 3, CpuAllocator>,
    prompt: String,
}

#[derive(Clone)]
struct MyResponse {
    result: String,
}

// Define lightweight metadata to avoid cloning heavy data
#[derive(Clone)]
struct MyMetadata {
    prompt: String,
    image_size: ImageSize,
}

impl RequestMetadata for MyRequest {
    type Metadata = MyMetadata;

    fn metadata(&self) -> Self::Metadata {
        MyMetadata {
            prompt: self.prompt.clone(),
            image_size: self.image.size(), // Only size, not the full image
        }
    }
}

// Implement your model
struct MyModel;

impl InfernumModel for MyModel {
    type Request = MyRequest;
    type Response = MyResponse;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn run(&mut self, request: Self::Request) -> Result<Self::Response, Self::Error> {
        // Your inference logic here
        Ok(MyResponse {
            result: format!("Processed: {}", request.prompt),
        })
    }
}
```

### 2. Create and Use the Engine

```rust
use kornia_infernum::{InfernumEngine, InfernumEngineResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the engine with your model
    let engine = InfernumEngine::new(MyModel);
    
    // Load an image
    let image = kornia_io::jpeg::read_image_jpeg_rgb8("path/to/image.jpg")?;
    
    // Create a request
    let request = MyRequest {
        image,
        prompt: "Describe this image".to_string(),
    };
    
    // Schedule inference (non-blocking)
    engine.schedule_inference(request);
    
    // Poll for results
    loop {
        match engine.try_poll_response() {
            InfernumEngineResult::Success(response) => {
                println!("Response: {}", response.response.result);
                println!("Duration: {:?}", response.duration);
                println!("Original prompt: {}", response.request_metadata.prompt);
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

## 🌐 Production Server Example

Kornia Infernum includes a production-ready HTTP server using PaliGemma:

```bash
cargo run --example infernum --features cuda  # With CUDA support
```

The server provides REST endpoints:

- 📤 `POST /inference` - Submit inference requests
- 📥 `GET /results` - Retrieve results with telemetry

Example request:
```json
{
  "image_path": "path/to/image.jpg",
  "prompt": "What do you see in this image?"
}
```

Example response:
```json
{
  "status": "success",
  "response": {
    "prompt": "What do you see in this image?",
    "start_time": 1234567890,
    "duration": "250ms",
    "response": "I can see a beautiful landscape with mountains..."
  }
}
```

## 🏗️ Architecture

### Core Components

- **`InfernumModel`** - Trait for implementing custom models
- **`RequestMetadata`** - Trait for extracting lightweight telemetry data
- **`InfernumEngine`** - High-performance inference engine with background processing
- **`InfernumEngineResponse`** - Rich response with telemetry and original request metadata

### Design Principles

1. **Performance First** - Avoid unnecessary cloning of heavy data like images
2. **Type Safety** - Fully generic design with compile-time guarantees
3. **Production Ready** - Built-in monitoring, error handling, and state management
4. **Flexibility** - Support any model through trait-based design

## 🔧 Requirements

- 🦀 Rust 2024 edition
- Optional: CUDA support for GPU acceleration

## 📊 Telemetry Features

- ⏱️ **Timing** - Precise inference duration tracking
- 🆔 **Request IDs** - Unique tracking for each inference
- 📝 **Metadata** - Lightweight request information without heavy data
- 🔄 **State Management** - Real-time engine state monitoring

## 📜 License

Licensed under the Apache License, Version 2.0.

## 👏 Acknowledgments

- Part of the [Kornia](https://github.com/kornia) ecosystem for computer vision in Rust
- Designed for production AI workloads with performance and monitoring in mind