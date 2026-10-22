"""Configuration constants for the bias annotation pipeline."""

# File paths (relative; run the pipeline from this directory).
DEFAULT_INPUT = "vldbench.jsonl"
DEFAULT_OUTPUT = "annotated_final.jsonl"

# Model and request settings.
MODEL = "gpt-5.5"
MAX_COMPLETION_TOKENS = 16000
MAX_RETRIES = 2
BASE_SLEEP = 10
MAX_WORKERS = 2

# Pipeline defaults.
DEFAULT_CALL_SLEEP = 5.0
DEFAULT_CHECKPOINT_EVERY = 20

# Annotation schema vocabularies.
VALID_BIAS_TYPES = [
    "loaded_language",
    "euphemism",
    "dehumanizing_language",
    "opinion_as_fact",
    "unsupported_generalization",
    "stereotypical_association",
    "sensationalism",
    "informational_bias",
]

VALID_SEGMENT_SEVERITY = ["Low", "Medium", "High"]
