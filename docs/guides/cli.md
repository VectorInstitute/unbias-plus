# Command line

Analyze a string:

```bash
unbias-plus --text "Women are too emotional to lead."
```

Analyze a file and emit JSON:

```bash
unbias-plus --file article.txt --json
```

Start the API server and demo UI. The default model is `vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-Legacy` and the default port is `8000`. The demo UI is served at the same host and port:

```bash
unbias-plus --serve
unbias-plus --serve --model path/to/model --port 8000
unbias-plus --serve --load-in-4bit   # reduce VRAM (optional)
```

**Available options:** `--model`, `--load-in-4bit`, `--max-new-tokens` (default `2048`), `--host`, `--port`, `--json`.
