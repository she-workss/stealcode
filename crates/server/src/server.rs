use core::time::Duration;
use std::{net::SocketAddr, path::Path, sync::Arc};

use axum::{
    Extension, Router,
    http::{Request, Response, header},
};
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

use crate::api;

const DEFAULT_PORT: u16 = 8000;

async fn shutdown_signal() -> anyhow::Result<()> {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )?
        .recv()
        .await;
        Ok(())
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        result = ctrl_c => {
            result?;
            info!("received Ctrl+C");
        }
        _ = terminate => info!("received SIGTERM"),
    }
    info!("starting graceful shutdown");
    Ok(())
}

pub async fn run_server(
    settings: &Settings,
    _project: Option<&Path>,
) -> anyhow::Result<()> {
    let port = settings
        .server
        .as_ref()
        .and_then(|s| s.port)
        .unwrap_or(DEFAULT_PORT);
    let api_routes = api::routes();
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
    let shutdown = async {
        if let Err(error) = shutdown_signal().await {
            tracing::error!(
                "failed to install shutdown signal handler: {error}"
            );
        }
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
