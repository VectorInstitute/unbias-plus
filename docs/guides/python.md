# Python API

```python
from unbias_plus import UnBiasPlus, BiasResult, BiasedSegment

pipe = UnBiasPlus()  # default: vector-institute/Qwen3-8B-UnBias-Plus-SFT-Instruct-V2
result = pipe.analyze("Women are too emotional to lead.")

# Result fields
print(result.binary_label)    # "biased" | "unbiased"
print(result.severity)         # 0-10 (article-level)
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

`UnBiasPlus()` and `UnBiasModel` both default to `max_new_tokens=8096`.

For the full `BiasResult` schema, see [How it works](../how_it_works.md#the-structured-output-is-the-point) or the [API Reference](../api.md).
