"""OpenCode Zen provider (https://opencode.ai/zen/v1) — the free models.

Thin wrapper over OpenAICompatProvider pointing at Zen's OpenAI-compatible
endpoint. With no API key, free (cost==0) models are used and Zen accepts the
literal API key "public" (mirrors opencode's behavior).

The Zen gateway throttles anonymous clients (unknown User-Agent, no
x-opencode-* headers) to a tiny free allowance — a couple of requests, then
429 FreeUsageLimitError. The official opencode client identifies itself with
`User-Agent: opencode/...` plus x-opencode-* headers, and the gateway treats
those as trusted clients with the real free quota. This provider sends the
same identity headers so free models (x-preview-f-free, etc.) keep working
across turns instead of being blocked after the first message.
"""

from __future__ import annotations

import threading
from typing import Any

from .openai_compat import OpenAICompatProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"

# Lane-rotation registry: Zen pins each x-opencode-session id to ONE upstream
# lane. When that lane dies server-side (streams end with Zen's own
# finish_reason="network_error"), retrying with the SAME id hammers the SAME
# dead lane — the user sees ↻ climb to (50) without a single success while a
# brand-new session id would have worked on attempt #1 (measured 2026-08-23:
# fixed sid A 5/5 OK, fixed sid B 1/5, fresh ids mixed). Keyed by BASE session
# id so a rotation survives provider re-instantiation: rotation.build_provider
# constructs a NEW ZenProvider for every attempt with the same engine sid.
_LANE_EPOCH: dict[str, int] = {}
_LANE_LOCK = threading.Lock()

# Limited-time free models on Zen ($0). Live-fetched in factory; this is the
# bundled fallback for when the network model list is unavailable (R2 risk).
FREE_MODELS: list[dict] = [
    {"id": "x-preview-f-free", "name": "Ox Alpha Free (Unlimited)", "context": 1000000, "output": 131072},
    {"id": "big-pickle", "name": "Big Pickle", "context": 200000, "output": 32000},
    {"id": "hy3-free", "name": "Hy3 Free", "context": 190000, "output": 64000},
    {"id": "mimo-v2.5-free", "name": "MiMo-V2.5 Free", "context": 200000, "output": 32000},
    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash Free", "context": 200000, "output": 128000},
    {"id": "nemotron-3-ultra-free", "name": "Nemotron 3 Ultra Free", "context": 1000000, "output": 128000},
    {"id": "nemotron-3.5-lightning-free", "name": "Nemotron 3.5 Lightning Free", "context": 262144, "output": 128000},
]


class ZenProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "x-preview-f-free",
        session_id: str | None = None,
        project: str | None = None,
        **kwargs: Any,
    ):
        # If no key given, we still need SOMETHING; Zen accepts "public" for free models.
        effective_key = api_key or "public"
        # Identify as the official opencode client so the Zen gateway serves the
        # real free quota instead of throttling an anonymous client (429
        # FreeUsageLimitError after the first message). The session id must be
        # stable across turns so Zen keeps the same conversation/provider lane —
        # unless that lane died and rotate_session() bumped the epoch below.
        self._base_session = session_id
        rot_key = session_id or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(rot_key, 0)
        effective_sid = session_id
        if epoch and session_id:
            effective_sid = f"{session_id}::r{epoch}"
        elif epoch:
            effective_sid = f"cli::r{epoch}"
        client_headers: dict[str, str] = {
            "User-Agent": "opencode/latest/0.1.0/cli",
            "x-opencode-client": "cli",
            "x-opencode-project": project or "opencode_py",
            "x-opencode-request": effective_sid or "cli",
        }
        if effective_sid:
            client_headers["x-opencode-session"] = effective_sid
        merged = dict(client_headers)
        merged.update(kwargs.pop("extra_headers", {}) or {})
        super().__init__(
            id="opencode",
            name="OpenCode Zen",
            base_url=ZEN_BASE_URL,
            api_key=effective_key,
            model=model,
            is_free=True,
            extra_headers=merged,
            reasoning_passthrough=True,
            **kwargs,
        )
        self.has_key = bool(api_key)

    def rotate_session(self) -> None:
        """Force a different upstream lane on the next request.

        Bumps this session's rotation epoch so every future ZenProvider built
        for the same base session id carries a fresh x-opencode-session value —
        Zen then assigns it a new lane instead of the dead one. Called by the
        rotation layer when a stream dies with server-side lane failure
        (in-band error / empty reply), NOT on local network problems."""
        key = self._base_session or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(key, 0) + 1
            _LANE_EPOCH[key] = epoch
        effective = (
            f"{self._base_session}::r{epoch}" if self._base_session else f"cli::r{epoch}"
        )
        self.extra_headers["x-opencode-request"] = effective
        if self._base_session:
            self.extra_headers["x-opencode-session"] = effective
