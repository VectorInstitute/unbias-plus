# Python API

```python
from unbias_plus import UnBiasPlus, BiasResult, BiasedSegment

pipe = UnBiasPlus()  # default: vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-Legacy
result = pipe.analyze("Women are too emotional to lead.")

# Result fields
print(result.binary_label)    # "biased" | "unbiased"
print(result.severity)         # 1-5 (article-level)
print(result.bias_found)       # bool
print(result.unbiased_text)   # full neutral rewrite

for seg in result.biased_segments:
    print(seg.original, seg.replacement, seg.severity, seg.bias_type, seg.reasoning)
    # seg.start, seg.end are character offsets in the original text

# Formatted outputs
cli_str  = pipe.analyze_to_cli("...")    # human-readable terminal output
data     = pipe.analyze_to_dict("...")   # plain dict
json_str = pipe.analyze_to_json("...")   # pretty-printed JSON string
```

`UnBiasPlus()` defaults to `max_new_tokens=4096`. The underlying `UnBiasModel` class defaults to `2048` when constructed directly.

For the full `BiasResult` schema, see [How it works](../how_it_works.md#the-structured-output-is-the-point) or the [API Reference](../api.md).
