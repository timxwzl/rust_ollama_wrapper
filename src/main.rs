use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, env, net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

// Define request/response structures
#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

// App state
struct AppState {
    client: Client,
    ollama_host: String,
    model_name: String,
}

#[tokio::main]
async fn main() {
    // Initialize environment
    dotenvy::dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Initialize state
    let state = Arc::new(AppState {
        client: Client::new(),
        ollama_host: env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string()),
        model_name: env::var("MODEL_NAME").unwrap_or_else(|_| "llama3.2:1b".to_string()),
    });

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/generate", post(generate))
        .route("/stream", post(stream))
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    info!("Starting server on {}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(&addr).await.unwrap(),
        app.into_make_service(),
    )
    .await
    .unwrap();
}

// Route handlers
async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "message": "Assistant is running!" }))
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (axum::http::StatusCode, String)> {
    info!("Sending prompt to Ollama at {}", state.ollama_host);
    info!("Using model: {}", state.model_name);
    info!("Prompt: {}", request.prompt);

    let target_url = format!("{}/api/generate", state.ollama_host);

    let payload = serde_json::json!({
        "model": state.model_name,
        "prompt": request.prompt,
        "stream": false
    });

    let response = state
        .client
        .post(&target_url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            error!("Error communicating with Ollama: {}", e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error communicating with Ollama: {}", e),
            )
        })?;

    info!("Ollama response status: {}", response.status());

    let ollama_response: OllamaResponse = response.json().await.map_err(|e| {
        error!("Error parsing Ollama response: {}", e);
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Invalid response format from Ollama".to_string(),
        )
    })?;

    info!("Successfully parsed response");

    Ok(Json(GenerateResponse {
        response: ollama_response.response,
    }))
}

async fn stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    info!("Starting streaming response");

    let target_url = format!("{}/api/generate", state.ollama_host);
    let client = state.client.clone();

    let stream = async_stream::stream! {
        let payload = serde_json::json!({
            "model": state.model_name,
            "prompt": request.prompt,
            "stream": true
        });

        let response = match client.post(&target_url).json(&payload).send().await {
            Ok(response) => response,
            Err(e) => {
                error!("Error communicating with Ollama: {}", e);
                yield Ok(Event::default().data(format!("Error: {}", e)));
                return;
            }
        };

        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                        match serde_json::from_str::<OllamaResponse>(&text) {
                            Ok(response) => {
                                info!(response.response);
                                yield Ok(Event::default().data(response.response));
                            },
                            Err(e) => {
                                error!("Error parsing response: {}", e);
                                continue;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Error reading stream: {}", e);
                    yield Ok(Event::default().data(format!("Error: {}", e)));
                    break;
                }
            }
        }
    };

    Sse::new(stream)
}
