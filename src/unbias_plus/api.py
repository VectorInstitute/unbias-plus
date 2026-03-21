"""FastAPI server for unbias-plus."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Generator, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from unbias_plus.model import DEFAULT_MODEL
from unbias_plus.parser import parse_llm_output
from unbias_plus.pipeline import UnBiasPlus
from unbias_plus.prompt import build_messages
from unbias_plus.schema import BiasResult, compute_offsets


DEMO_DIR = Path(__file__).parent / "demo"

# When set, the demo app acts as a thin proxy to a remote vLLM endpoint
# (OpenAI-compatible API). No local model is loaded.
# Example: https://unbias-plus-vllm-xxxx.us-central1.run.app/v1
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "unbias-plus")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "1000"))
# Tune via RATE_LIMIT env var, e.g. "20/minute", "100/hour"
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")


def _get_client_ip(request: Request) -> str:
    """Return the real client IP, respecting Cloud Run's X-Forwarded-For header."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Cloud Run prepends the real client IP as the first entry
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_get_client_ip)


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
        # Remote vLLM — no local model load needed.
        app.state.vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        app.state.pipe = None
        print(f"Using remote vLLM at {VLLM_BASE_URL} (model: {VLLM_MODEL_NAME})")
    else:
        app.state.vllm_client = None
        model_path = getattr(app.state, "model_name_or_path", DEFAULT_MODEL)
        load_in_4bit = getattr(app.state, "load_in_4bit", False)
        app.state.pipe = UnBiasPlus(
            model_name_or_path=model_path,
            load_in_4bit=load_in_4bit,
        )
        # Warmup: run a short dummy inference to compile CUDA kernels so the
        # first real user request doesn't pay the JIT compilation penalty (~60s).
        print("Warming up CUDA kernels...")
        try:
            app.state.pipe.analyze("Warmup.")
            print("Warmup complete.")
        except Exception:
            pass  # warmup failure is non-fatal

    yield
    app.state.pipe = None
    app.state.vllm_client = None


app = FastAPI(
    title="unbias-plus API",
    description="Bias detection and debiasing: identify segments, classify severity, reasoning and replacements, full neutral rewrite.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Serve demo static files if demo directory exists
if (DEMO_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=DEMO_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the demo UI.

    Returns
    -------
    str
        HTML content of the demo page.

    Raises
    ------
    HTTPException
        404 if the demo directory is not found.

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
@limiter.limit(RATE_LIMIT)
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
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = completion.choices[0].message.content or ""
            result = parse_llm_output(raw)
            segments = compute_offsets(body.text, result.biased_segments)
            return result.model_copy(
                update={"biased_segments": segments, "original_text": body.text}
            )
        assert (
            pipe is not None
        )  # guaranteed: checked above that at least one backend is set
        return cast(BiasResult, pipe.analyze(body.text))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/analyze/stream")
@limiter.limit(RATE_LIMIT)
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
                # Stream via vLLM's OpenAI-compatible SSE endpoint.
                # vLLM handles concurrency and continuous batching server-side.
                stream = vllm_client.chat.completions.create(
                    model=VLLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0,
                    stream=True,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        raw_output += token
                        yield "data: " + json.dumps({"t": token}) + "\n\n"
            else:
                # Local model streaming via HuggingFace TextIteratorStreamer.
                assert pipe is not None  # guaranteed: checked above
                for token in pipe._model.generate_stream(messages):
                    raw_output += token
                    yield "data: " + json.dumps({"t": token}) + "\n\n"

            result = parse_llm_output(raw_output)
            segments_with_offsets = compute_offsets(text, result.biased_segments)
            final = result.model_copy(
                update={
                    "biased_segments": segments_with_offsets,
                    "original_text": text,
                }
            )
            yield (
                "data: "
                + json.dumps({"result": json.loads(final.model_dump_json())})
                + "\n\n"
            )
        except ValueError as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def serve(
    model_name_or_path: str = "vector-institute/Qwen3-8B-UnBias-Plus-SFT",
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
    model_name_or_path : str
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
