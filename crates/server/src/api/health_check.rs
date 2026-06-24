use axum::{Json, Router, routing::get};
use serde_json::{Value, json};

pub(crate) fn routes() -> Router {
    Router::new().route("/", get(health_check))
}

pub(crate) async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
    }))
}
