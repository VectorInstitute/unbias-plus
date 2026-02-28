"""FastAPI server for unbias-plus."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from unbias_plus.model import DEFAULT_MODEL
from unbias_plus.pipeline import UnBiasPlus
from unbias_plus.schema import BiasResult


_pipe: UnBiasPlus | None = None
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
    global _pipe
    model_path = getattr(app.state, "model_name_or_path", DEFAULT_MODEL)
    load_in_4bit = getattr(app.state, "load_in_4bit", False)
    _pipe = UnBiasPlus(
        model_name_or_path=model_path,
        load_in_4bit=load_in_4bit,
    )
    yield
    _pipe = None


app = FastAPI(
    title="unbias-plus API",
    description="Bias detection and debiasing using a single LLM.",
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
def health() -> HealthResponse:
    """Check if the server and model are ready.

    Returns
    -------
    HealthResponse
        Server status and loaded model name.

    """
    return HealthResponse(
        status="ok",
        model=str(_pipe._model.model_name_or_path) if _pipe else "not loaded",
    )


@app.post("/analyze", response_model=BiasResult)
def analyze(request: AnalyzeRequest) -> BiasResult:
    """Analyze input text for bias.

    Parameters
    ----------
    request : AnalyzeRequest
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
    if _pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        return _pipe.analyze(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


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
    >>> serve("VectorInstitute/unbias-plus-llama3", port=8000)

    """
    app.state.model_name_or_path = str(model_name_or_path)
    app.state.load_in_4bit = load_in_4bit
    print(f"Starting unbias-plus server at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)