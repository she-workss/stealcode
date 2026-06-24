use axum::{Json, Router, http::StatusCode};
use serde_json::Value;

pub(crate) mod health_check;

/// Routes for the API.
pub(crate) fn routes() -> Router {
    let health_check_routes =
        Router::new().nest("/health_check", health_check::routes());
    Router::new()
        .merge(health_check_routes)
        .fallback(api_fallback)
}

/// Fallback route for the API.
async fn api_fallback() -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "status": "Not Found" })),
    )
}
