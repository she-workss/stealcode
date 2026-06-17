use core::time::Duration;
use std::{net::SocketAddr, sync::Arc};

use axum::{
    Extension, Json, Router,
    http::{Request, Response, StatusCode, header},
};
use serde_json::Value;
use settings::{Settings, state::AppState};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    catch_panic::CatchPanicLayer,
    classify::ServerErrorsFailureClass,
    cors::{self, CorsLayer},
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    sensitive_headers::SetSensitiveRequestHeadersLayer,
    trace::TraceLayer,
};
use tracing::{Span, debug, info, info_span};

const DEFAULT_PORT: u16 = 8000;

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
        .expect("failed to install SIGTERM handler")
        .recv()
        .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => info!("received Ctrl+C"),
        () = terminate => info!("received SIGTERM"),
    }
    info!("starting graceful shutdown");
}

pub async fn run_server(settings: &Settings) -> anyhow::Result<()> {
    let port = settings
        .server
        .as_ref()
        .and_then(|s| s.port)
        .unwrap_or(DEFAULT_PORT);
    let api_routes = routes();
    let cors = CorsLayer::new()
        .allow_origin(cors::Any)
        .allow_methods(cors::Any)
        .allow_headers(cors::Any);
    let shared_state = Arc::new(AppState::default());
    let shared_settings = Arc::new(settings.clone());
    let sensitive_headers: Arc<[_]> = Arc::new([
        header::AUTHORIZATION,
        header::COOKIE,
        header::PROXY_AUTHORIZATION,
        header::SET_COOKIE,
    ]);
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request<_>| {
            let request_id = request
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("-");
            info_span!(
                "http.request",
                method = %request.method(),
                uri = %request.uri(),
                version = ?request.version(),
                request_id = %request_id,
            )
        })
        .on_request(())
        .on_response(
            |response: &Response<_>, latency: Duration, _span: &Span| {
                info!(
                    status = response.status().as_u16(),
                    latency_ms = latency.as_millis(),
                    "request completed"
                );
            },
        )
        .on_body_chunk(())
        .on_eos(())
        .on_failure(
            |_error: ServerErrorsFailureClass,
             _latency: Duration,
             _span: &Span| {
                debug!("something went wrong");
            },
        );
    let layers = ServiceBuilder::new()
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .layer(trace_layer)
        .layer(cors)
        .layer(CatchPanicLayer::new())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetSensitiveRequestHeadersLayer::from_shared(Arc::clone(
            &sensitive_headers,
        )))
        .layer(Extension(shared_state))
        .layer(Extension(shared_settings));
    let app = Router::new().nest("/api/v1", api_routes).layer(layers);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("server starting on port {port}");
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

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
