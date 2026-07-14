# REST API

Start the server with `unbias-plus --serve`. The demo web UI is at `http://localhost:8000/`; the same host and port serve the API.

| Endpoint | Description |
|----------|-------------|
| **GET /health** | Returns `{"status": "ok", "model": "<model_name_or_path>"}`. |
| **POST /analyze** | Body: `{"text": "Your text here"}`. Returns a JSON object matching the `BiasResult` schema. |

Example with `curl`:

```bash
curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Women are too emotional to lead."}'
```

Programmatic server start:

```python
from unbias_plus.api import serve

serve()  # default model, port 8000
# Or: serve("path/to/model", port=8000, load_in_4bit=False)
```

The default model is `vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-V2` (see `DEFAULT_MODEL` in `unbias_plus.model`).
