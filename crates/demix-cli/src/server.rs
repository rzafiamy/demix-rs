//! HTTP API.
//!
//! - `POST /v1/audio/separations` (multipart): `file` (required, any format
//!   `demix separate` reads), `stem` (optional: one stem name, or several
//!   comma-separated, or `all`), `format` (`s16` default, or `f32`),
//!   `model` (ignored: one model per process). One stem -> `audio/wav`;
//!   several or none -> `application/zip` of `<stem>.wav` files.
//! - `GET /health`, `GET /v1/models` (with the model's stem names).
//!
//! One separation at a time: requests queue on a mutex.

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use demix::Separator;
use demix::audio::{self, SampleFormat};
use serde_json::json;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

pub struct AppState {
    sep: Arc<Mutex<Separator>>,
    model_id: String,
    stems: Vec<String>,
}

type Shared = Arc<AppState>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = json!({"error": {"message": self.1, "type": kind}});
        (self.0, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

pub async fn serve(sep: Separator, model_id: String, host: &str, port: u16) -> Result<()> {
    let stems = sep.stem_names();
    let state = Arc::new(AppState {
        sep: Arc::new(Mutex::new(sep)),
        model_id,
        stems,
    });
    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/audio/separations", post(separations))
        .layer(DefaultBodyLimit::max(1 << 30))
        .with_state(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{addr}");
    eprintln!("demix server on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn models(State(st): State<Shared>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "owned_by": "demix",
            "stems": st.stems,
        }]
    }))
}

struct Request {
    file: Vec<u8>,
    filename: Option<String>,
    stems: Vec<String>,
    format: SampleFormat,
}

async fn parse_multipart(
    headers: &HeaderMap,
    body: Bytes,
    known: &[String],
) -> Result<Request, ApiError> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let boundary = multer::parse_boundary(ct)
        .map_err(|_| bad("expected multipart/form-data with a `file` field"))?;
    let stream = futures_util::stream::once(async move { Ok::<_, std::io::Error>(body) });
    let mut mp = multer::Multipart::new(stream, boundary);
    let mut file = None;
    let mut filename = None;
    let mut stem = String::new();
    let mut format = SampleFormat::S16;
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| bad(format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(String::from);
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| bad(format!("file: {e}")))?
                        .to_vec(),
                );
            }
            "stem" | "stems" => stem = field.text().await.map_err(|e| bad(e.to_string()))?,
            "format" | "sample_format" => {
                let v = field.text().await.map_err(|e| bad(e.to_string()))?;
                format = v
                    .trim()
                    .parse()
                    .map_err(|e: anyhow::Error| bad(e.to_string()))?;
            }
            _ => {} // model, response_format…: ignored
        }
    }
    let file = file
        .filter(|f| !f.is_empty())
        .ok_or_else(|| bad("missing `file`"))?;
    let stem = stem.trim();
    let stems: Vec<String> = if stem.is_empty() || stem.eq_ignore_ascii_case("all") {
        known.to_vec()
    } else {
        stem.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    for s in &stems {
        if !known.contains(s) {
            return Err(bad(format!(
                "unknown stem '{s}' (this model: {})",
                known.join(", ")
            )));
        }
    }
    Ok(Request {
        file,
        filename,
        stems,
        format,
    })
}

async fn separations(
    State(st): State<Shared>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let req = parse_multipart(&headers, body, &st.stems).await?;
    let ext = req
        .filename
        .as_deref()
        .and_then(|f| f.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()));
    let base = req
        .filename
        .as_deref()
        .and_then(|f| {
            f.rsplit_once('.')
                .map(|(b, _)| b.to_string())
                .or(Some(f.to_string()))
        })
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "audio".into());

    let guard = st.sep.clone().lock_owned().await;
    let wanted = req.stems.clone();
    let label = wanted.join(", ");
    let fmt = req.format;
    let (outputs, secs, elapsed) = tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
        let t = Instant::now();
        let a = audio::decode(req.file, ext.as_deref()).map_err(|e| bad(format!("{e:#}")))?;
        let secs = a.duration_secs();
        let mix = audio::conform(a, guard.cfg.channels(), guard.cfg.sample_rate)?;
        let stems = guard.separate(&mix, |_, _| {})?;
        let rate = guard.cfg.sample_rate;
        let mut out = Vec::new();
        for s in stems {
            if wanted.contains(&s.name) {
                out.push((s.name.clone(), audio::encode_wav(&s.audio, rate, fmt)?));
            }
        }
        Ok((out, secs, t.elapsed().as_secs_f64()))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {e}")))??;
    tracing::info!(
        "separated {secs:.1}s of audio in {elapsed:.1}s ({:.1}x realtime): {}",
        secs / elapsed.max(1e-9),
        label
    );

    let mut resp = if outputs.len() == 1 {
        let (name, wav) = outputs.into_iter().next().unwrap();
        let mut r = wav.into_response();
        r.headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
        set_disposition(&mut r, &format!("{base}_{name}.wav"));
        r
    } else {
        let zip = build_zip(&outputs)?;
        let mut r = zip.into_response();
        r.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/zip"),
        );
        set_disposition(&mut r, &format!("{base}_stems.zip"));
        r
    };
    if let Ok(v) = HeaderValue::from_str(&format!("{secs:.3}")) {
        resp.headers_mut().insert("x-audio-duration", v);
    }
    Ok(resp)
}

fn set_disposition(r: &mut Response, filename: &str) {
    // ASCII only: other characters become '_'
    let safe: String = filename
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() && c != '"' && c != '\\' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{safe}\"")) {
        r.headers_mut().insert(header::CONTENT_DISPOSITION, v);
    }
}

fn build_zip(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut cur = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut cur);
        // WAV barely compresses: store
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(files.iter().map(|(_, b)| b.len()).sum::<usize>() > u32::MAX as usize / 2);
        for (name, bytes) in files {
            z.start_file(format!("{name}.wav"), opts)?;
            z.write_all(bytes)?;
        }
        z.finish()?;
    }
    Ok(cur.into_inner())
}
