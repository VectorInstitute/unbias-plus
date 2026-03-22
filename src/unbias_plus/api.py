"""FastAPI server for unbias-plus."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Generator, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from unbias_plus.model import DEFAULT_MODEL
from unbias_plus.parser import parse_llm_output
from unbias_plus.pipeline import UnBiasPlus
from unbias_plus.prompt import build_messages
from unbias_plus.schema import BiasResult, compute_offsets


DEMO_DIR = Path(__file__).parent / "demo"


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
    model_path = getattr(app.state, "model_name_or_path", DEFAULT_MODEL)
    load_in_4bit = getattr(app.state, "load_in_4bit", False)
    app.state.pipe = UnBiasPlus(
        model_name_or_path=model_path,
        load_in_4bit=load_in_4bit,
    )
    yield
    app.state.pipe = None


app = FastAPI(
    title="unbias-plus API",
    description="Bias detection and debiasing: identify segments, classify severity, reasoning and replacements, full neutral rewrite.",
    version="0.1.0",
    lifespan=lifespan,
)

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
    pipe = getattr(request.app.state, "pipe", None)
    return HealthResponse(
        status="ok",
        model=str(pipe._model.model_name_or_path) if pipe else "not loaded",
    )


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
        500 if the model is not loaded or inference fails.
    HTTPException
        422 if the model output cannot be parsed.
    """
    pipe = getattr(request.app.state, "pipe", None)
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        return cast(BiasResult, pipe.analyze(body.text))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/analyze/stream")
def analyze_stream(request: Request, body: AnalyzeRequest) -> StreamingResponse:
    """Stream bias analysis tokens via SSE, then emit the final parsed result.

    Runs model generation in a background thread via TextIteratorStreamer.
    Each SSE event is a JSON object:

    - ``{"t": "<token>"}``     — one chunk per model generation step.
    - ``{"result": {...}}``    — final event with the full BiasResult.
    - ``{"error": "<msg>"}``   — emitted if inference or parsing fails.

    Parameters
    ----------
    request : Request
        FastAPI request (for app state).
    body : AnalyzeRequest
        Request body containing the text to analyze.

    Returns
    -------
    StreamingResponse
        Server-sent events stream with Content-Type text/event-stream.

    Raises
    ------
    HTTPException
        500 if the model is not loaded.
    """
    pipe = getattr(request.app.state, "pipe", None)
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    text = body.text

    def event_stream() -> Generator[str, None, None]:
        try:
            messages = build_messages(text)
            raw_output = ""

            # Stream tokens from the background generation thread
            for token in pipe._model.generate_stream(messages):
                raw_output += token
                yield "data: " + json.dumps({"t": token}) + "\n\n"

            # Full output accumulated — parse and compute offsets
            result = parse_llm_output(raw_output)
            segments = compute_offsets(text, result.biased_segments)
            final = result.model_copy(
                update={
                    "biased_segments": segments,
                    "original_text": text,
                }
            )
            yield (
                "data: "
                + json.dumps({"result": final.model_dump(mode="json")})
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
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


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
