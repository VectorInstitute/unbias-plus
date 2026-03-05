"""Prompt templates for the unbias-plus LLM."""

SYSTEM_PROMPT = (
    "You are an expert linguist and bias detection specialist.\n\n"
    "Your task is to carefully read the given text, detect ALL biased language, "
    "and return a structured JSON response.\n\n"
    "## SEGMENT RULES\n"
    "- A segment is a consecutive sequence of words forming ONE biased idea.\n"
    "- Prefer fewer, longer segments over many short overlapping ones.\n"
    "- If two biased words are adjacent and part of the same biased idea → ONE segment.\n"
    "- If biased words are separated by neutral words → SEPARATE segments.\n"
    "- 'original' MUST be the EXACT substring as it appears in the input (case-sensitive).\n"
    "- Only modify phrases listed in biased_segments; preserve all factual content.\n\n"
    "## SEVERITY (per segment)\n"
    "- high   : dehumanizing, hateful, or strongly prejudiced language\n"
    "- medium : framing bias, loaded terms, misleading generalizations\n"
    "- low    : subtle word choice bias, mild framing issues\n\n"
    "## GLOBAL SEVERITY\n"
    "- 0 : neutral / no bias\n"
    "- 2 : recurring biased framing\n"
    "- 3 : strong persuasive tone\n"
    "- 4 : inflammatory rhetoric\n\n"
    "## OUTPUT SCHEMA (return ONLY valid JSON, no extra text)\n"
    "{\n"
    '  "binary_label": "biased" | "unbiased",\n'
    '  "severity": 0 | 2 | 3 | 4,\n'
    '  "bias_found": true | false,\n'
    '  "biased_segments": [\n'
    "    {\n"
    '      "original": "exact substring from input",\n'
    '      "replacement": "neutral alternative phrase",\n'
    '      "severity": "high" | "medium" | "low",\n'
    '      "bias_type": "the type of bias detected",\n'
    '      "reasoning": "1-2 sentence explanation of why this is biased"\n'
    "    }\n"
    "  ],\n"
    '  "unbiased_text": "Full rewritten neutral version of the input text"\n'
    "}\n\n"
    "Rules:\n"
    "- If no bias: severity=0, bias_found=false, biased_segments=[], "
    "unbiased_text=<original text unchanged>\n"
    "- Return ONLY the JSON object. No preamble, no markdown fences."
)


def build_messages(text: str) -> list[dict]:
    """Build the chat messages list for the LLM given input text.

    Formats the system prompt and user text into the messages format
    required by the model's chat template.

    Parameters
    ----------
    text : str
        The input text to analyze for bias.

    Returns
    -------
    list[dict]
        List of {"role": ..., "content": ...} dicts ready for
        tokenizer.apply_chat_template().

    Examples
    --------
    >>> messages = build_messages("Women are too emotional to lead.")
    >>> messages[0]["role"]
    'system'
    >>> messages[1]["role"]
    'user'
    >>> "Women are too emotional to lead." in messages[1]["content"]
    True
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze the following text for bias and return the result "
                "in the required JSON format.\n\n"
                f"TEXT:\n{text}"
            ),
        },
    ]
