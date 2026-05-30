"""FastAPI server for unbias-plus."""

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from unbias_plus.model import DEFAULT_MODEL
from unbias_plus.parser import parse_llm_output
from unbias_plus.pipeline import UnBiasPlus, finalize_result
from unbias_plus.prompt import build_messages
from unbias_plus.schema import BiasResult


DEMO_DIR = Path(__file__).parent / "demo"

# When set, the demo app acts as a thin proxy to a remote vLLM endpoint
# (OpenAI-compatible API). No local model is loaded.
# In production this points to proxy.vectorinstitute.ai/v1.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "unbias-plus")
# Service-level key for the Vector proxy gateway — never exposed to end users.
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "5000"))
# Cloud project for BigQuery feedback storage (auto-set by Cloud Run).
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "unbias-toolkit")
_BQ_DATASET = "unbias_plus"
_BQ_TABLE = "feedback"

_bq_lock = threading.Lock()
_bq_cache: dict[str, Any] = {}


def _safe_error(e: Exception) -> str:
    """Return str(e) with the proxy API key redacted."""
    msg = str(e)
    if VLLM_API_KEY and VLLM_API_KEY != "EMPTY":
        msg = msg.replace(VLLM_API_KEY, "[REDACTED]")
    return msg


def _get_bq_client() -> Any:
    """Return a cached BigQuery client, creating the dataset/table on first call."""
    if "client" not in _bq_cache:
        with _bq_lock:
            if "client" not in _bq_cache:
                from google.cloud import bigquery  # noqa: PLC0415

                client = bigquery.Client(project=GCP_PROJECT)
                _ensure_bq_table(client)
                _bq_cache["client"] = client
    return _bq_cache["client"]


def _ensure_bq_table(client: Any) -> None:
    """Create the BigQuery feedback dataset and table if they don't exist."""
    from google.cloud import bigquery  # noqa: PLC0415

    dataset_id = f"{GCP_PROJECT}.{_BQ_DATASET}"
    try:
        client.get_dataset(dataset_id)
    except Exception:
        client.create_dataset(dataset_id)

    table_id = f"{GCP_PROJECT}.{_BQ_DATASET}.{_BQ_TABLE}"
    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("reaction", "STRING"),
            bigquery.SchemaField("message", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("input_text", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("rating", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("speed", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("accuracy", "STRING", mode="NULLABLE"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))


class FeedbackRequest(BaseModel):
    """Request body for the feedback endpoint."""

    reaction: str  # "like" or "dislike" (required)
    message: str = ""  # optional free-text comment
    input_text: str = ""  # the text that was analyzed
    rating: int | None = None  # 1–5 star rating
    speed: str | None = None  # "too_slow" | "acceptable" | "fast"
    accuracy: str | None = None  # "not_accurate" | "somewhat" | "very_accurate"


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint.

    Attributes
    ----------
    text : str
        The input text to analyze for bias.
    """

    text: str


class HealthResponse(BaseModel):
    """Response body for the health endpoint.

    Attributes
    ----------
    status : str
        Server status string.
    model : str
        Currently loaded model name or path.
    """

    status: str
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup and release on shutdown.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Yields
    ------
    None
    """
    if VLLM_BASE_URL:
        from openai import OpenAI  # noqa: PLC0415

        app.state.vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        app.state.pipe = None
        print(f"Using remote vLLM via {VLLM_BASE_URL} (model: {VLLM_MODEL_NAME})")
    else:
        app.state.vllm_client = None
        model_path = getattr(app.state, "model_name_or_path", DEFAULT_MODEL)
        load_in_4bit = getattr(app.state, "load_in_4bit", False)
        app.state.pipe = UnBiasPlus(
            model_name_or_path=model_path,
            load_in_4bit=load_in_4bit,
        )
        pipe_ref = app.state.pipe

        def _cuda_warmup() -> None:
            print("Warming up CUDA kernels (background)...")
            try:
                pipe_ref.analyze("Warmup.")
                print("Warmup complete.")
            except Exception:
                pass  # warmup failure is non-fatal

        threading.Thread(target=_cuda_warmup, daemon=True).start()

    yield
    app.state.pipe = None
    app.state.vllm_client = None


app = FastAPI(
    title="unbias-plus API",
    description="Bias detection and debiasing: identify segments, classify severity, reasoning and replacements, full neutral rewrite.",
    version="0.1.0",
    lifespan=lifespan,
)

if (DEMO_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=DEMO_DIR / "static"), name="static")

FAVICON_PATH = DEMO_DIR / "static" / "favicon-48x48.svg"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the demo favicon for browsers that request /favicon.ico by default."""
    if not FAVICON_PATH.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(FAVICON_PATH, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse, response_model=None)
def index() -> str:
    """Serve the landing page in cloud mode, or the demo UI locally.

    Returns
    -------
    str
        HTML content.

    Raises
    ------
    HTTPException
        404 if the template is not found.
    """
    if VLLM_BASE_URL:
        html_file = DEMO_DIR / "templates" / "landing.html"
        if html_file.exists():
            return html_file.read_text()
    html_file = DEMO_DIR / "templates" / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found.")
    return html_file.read_text()


@app.get("/demo", response_class=HTMLResponse, response_model=None)
def demo_page() -> str:
    """Serve the demo UI.

    Returns
    -------
    str
        index.html content.

    Raises
    ------
    HTTPException
        404 if index.html is not found.
    """
    html_file = DEMO_DIR / "templates" / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found.")
    return html_file.read_text()


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Check if the server and model are ready.

    Returns
    -------
    HealthResponse
        Server status and loaded model name.
    """
    vllm_client = getattr(request.app.state, "vllm_client", None)
    pipe = getattr(request.app.state, "pipe", None)
    if vllm_client is not None:
        return HealthResponse(
            status="ok", model=f"{VLLM_MODEL_NAME} (vLLM @ {VLLM_BASE_URL})"
        )
    if pipe is not None:
        return HealthResponse(status="ok", model=str(pipe._model.model_name_or_path))
    return HealthResponse(status="starting", model="not loaded")


@app.post("/analyze", response_model=BiasResult)
def analyze(request: Request, body: AnalyzeRequest) -> BiasResult:
    """Analyze input text for bias.

    Parameters
    ----------
    request : Request
        FastAPI request (for app state).
    body : AnalyzeRequest
        Request body containing the text to analyze.

    Returns
    -------
    BiasResult
        Structured bias analysis result with character offsets.

    Raises
    ------
    HTTPException
        500 if no model backend is available or inference fails.
    HTTPException
        422 if the input is too long or output cannot be parsed.

    """
    vllm_client = getattr(request.app.state, "vllm_client", None)
    pipe = getattr(request.app.state, "pipe", None)
    if vllm_client is None and pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if len(body.text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Input too long: {len(body.text)} chars (max {MAX_INPUT_CHARS}).",
        )
    try:
        if vllm_client is not None:
            completion = vllm_client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=build_messages(body.text),
                max_tokens=4096,
                temperature=0,
                stop=["<|im_end|>", "<|endoftext|>"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stop_token_ids": [151645, 151643],
                },
            )
            raw = completion.choices[0].message.content or ""
            result = parse_llm_output(raw)
            return finalize_result(body.text, result)
        assert pipe is not None
        return cast(BiasResult, pipe.analyze(body.text))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=_safe_error(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=_safe_error(e)) from e


def _sse_result_line_or_none(raw_output: str, original_text: str) -> str | None:
    """Return the final SSE ``data: {"result":...}`` line if *raw_output* is complete.

    Cheap guard first (trailing ``}``), then :func:`parse_llm_output`. Used to stop
    streaming as soon as the model has finished the JSON instead of consuming tokens
    until ``max_tokens``.
    """
    if not raw_output.rstrip().endswith("}"):
        return None
    try:
        result = parse_llm_output(raw_output)
    except ValueError:
        return None
    final = finalize_result(original_text, result)
    return (
        "data: " + json.dumps({"result": json.loads(final.model_dump_json())}) + "\n\n"
    )


@app.post("/analyze/stream")
def analyze_stream(request: Request, body: AnalyzeRequest) -> StreamingResponse:
    """Stream bias analysis tokens via SSE, then emit the final parsed result.

    Parameters
    ----------
    request : Request
        FastAPI request (for app state).
    body : AnalyzeRequest
        Request body containing the text to analyze.

    Returns
    -------
    StreamingResponse
        Server-sent events stream. Each event is a JSON object:
        - ``{"t": "<token>"}`` for each generated token.
        - ``{"result": {...}}`` as the final event with the full BiasResult.
          Emitted as soon as the accumulated output parses as a full result
          (typically right after the closing ``}`` of the JSON), so the stream
          can end before ``max_tokens`` if the model finishes the object.
        - ``{"error": "<message>"}`` if inference fails.

    """
    vllm_client = getattr(request.app.state, "vllm_client", None)
    pipe = getattr(request.app.state, "pipe", None)
    if vllm_client is None and pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if len(body.text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Input too long: {len(body.text)} chars (max {MAX_INPUT_CHARS}).",
        )

    text = body.text

    def event_stream() -> Generator[str, None, None]:
        try:
            messages = build_messages(text)
            raw_output = ""

            if vllm_client is not None:
                stream = vllm_client.chat.completions.create(
                    model=VLLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                    stream=True,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "stop_token_ids": [151645, 151643],
                    },
                )
                try:
                    for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        if token:
                            raw_output += token
                            yield "data: " + json.dumps({"t": token}) + "\n\n"
                        early = _sse_result_line_or_none(raw_output, text)
                        if early is not None:
                            yield early
                            return
                finally:
                    stream.close()
            else:
                assert pipe is not None
                for token in pipe._model.generate_stream(messages):
                    raw_output += token
                    yield "data: " + json.dumps({"t": token}) + "\n\n"
                    early = _sse_result_line_or_none(raw_output, text)
                    if early is not None:
                        yield early
                        return

            result = parse_llm_output(raw_output)
            final = finalize_result(text, result)
            yield (
                "data: "
                + json.dumps({"result": json.loads(final.model_dump_json())})
                + "\n\n"
            )
        except ValueError as e:
            yield "data: " + json.dumps({"error": _safe_error(e)}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"error": _safe_error(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest) -> dict[str, Any]:
    """Save user feedback to BigQuery.

    Returns
    -------
    dict
        ``{"ok": True}`` on success.

    Raises
    ------
    HTTPException
        404 in local mode (no VLLM_BASE_URL).
        422 if reaction value is invalid.
        500 if the BigQuery write fails.
    """
    if not VLLM_BASE_URL:
        raise HTTPException(status_code=404, detail="Not available in local mode.")

    if body.reaction not in ("like", "dislike"):
        raise HTTPException(
            status_code=422, detail="reaction must be 'like' or 'dislike'"
        )

    try:
        bq = _get_bq_client()
        table_id = f"{GCP_PROJECT}.{_BQ_DATASET}.{_BQ_TABLE}"
        rows = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reaction": body.reaction,
                "message": body.message or None,
                "input_text": body.input_text[:500] if body.input_text else None,
                "rating": body.rating,
                "speed": body.speed or None,
                "accuracy": body.accuracy or None,
            }
        ]
        errors = bq.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback: {e}"
        ) from e

    return {"ok": True}


def serve(
    model_name_or_path: str | Path = DEFAULT_MODEL,
    host: str = "0.0.0.0",
    port: int = 8000,
    load_in_4bit: bool = False,
    reload: bool = False,
) -> None:
    """Start the unbias-plus API server with the demo UI.

    Loads the model and starts a uvicorn server. The demo UI
    is served at http://localhost:{port}/ and the API is at
    http://localhost:{port}/analyze.

    Parameters
    ----------
    model_name_or_path : str | Path
        HuggingFace model ID or local path to the model.
    host : str
        Host address to bind to. Default is '0.0.0.0'.
    port : int
        Port to listen on. Default is 8000.
    load_in_4bit : bool
        Load model in 4-bit quantization. Default is False.
    reload : bool
        Enable auto-reload on code changes. Default is False.

    Examples
    --------
    >>> from unbias_plus.api import serve
    >>> serve("Qwen/Qwen3-4B", port=8000)  # doctest: +SKIP

    """
    app.state.model_name_or_path = str(model_name_or_path)
    app.state.load_in_4bit = load_in_4bit
    print(f"Starting unbias-plus server at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)
