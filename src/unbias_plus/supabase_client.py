"""Supabase client singleton.

Cloud only — never imported by the PyPI package.
Uses the service role key for privileged server-side writes.
"""

import os

from supabase import Client, create_client


_cache: dict[str, Client] = {}


def get_supabase() -> Client:
    """Return a cached Supabase client. Initialised once on first call."""
    if "client" not in _cache:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _cache["client"] = create_client(url, key)
    return _cache["client"]
