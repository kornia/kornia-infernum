use argh::FromArgs;
use std::path::PathBuf;

mod messages;

// defaults for the client
const DEFAULT_HOST: &str = "localhost";
const DEFAULT_PORT: u16 = 3000;

#[derive(FromArgs)]
/// Infernum client for running inference and checking results
struct ClientArgs {
    /// the host to connect to
    #[argh(option, short = 'h', default = "DEFAULT_HOST.to_string()")]
    host: String,

    /// the port to connect to
    #[argh(option, short = 'p', default = "DEFAULT_PORT")]
    port: u16,

    /// command to execute: "inference" or "results"
    #[argh(subcommand)]
    command: ClientCommands,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum ClientCommands {
    Inference(InferenceCommand),
    Results(ResultsCommand),
}

#[derive(FromArgs)]
/// Run inference with an image and prompt
#[argh(subcommand, name = "inference")]
struct InferenceCommand {
    /// the path to the image
    #[argh(option, short = 'i')]
    image_path: PathBuf,

    /// the prompt to use
    #[argh(option, short = 'p')]
    prompt: String,
}

#[derive(FromArgs)]
/// Check inference results
#[argh(subcommand, name = "results")]
struct ResultsCommand {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: ClientArgs = argh::from_env();

    let client = reqwest::Client::new();

    // format the host and port
    let addr = format!("{}:{}", args.host, args.port);

    match args.command {
        ClientCommands::Inference(inference_command) => {
            let response = client
                .post(format!("http://{}/inference", addr))
                .json(&messages::InferenceRequest {
                    image_path: inference_command.image_path,
                    prompt: inference_command.prompt,
                })
                .send()
                .await?;

            let result = response.json::<serde_json::Value>().await?;
            println!("Result: {}", serde_json::to_string_pretty(&result)?);
        }
        ClientCommands::Results(_) => {
            let response = client
                .get(format!("http://{}/results", addr))
                .send()
                .await?;

            let result = response.json::<serde_json::Value>().await?;
            println!("Result: {}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(())
}
