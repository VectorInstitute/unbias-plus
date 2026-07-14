"""OpenAI client construction."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """Build an OpenAI client from ``OPENAI_API_KEY`` (and optional base URL)."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)
