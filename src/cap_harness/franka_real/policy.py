"""The bench's learned-policy client: one observation out, one action chunk back.

The policy itself runs in its own process (``policy_server.py`` in the hebbot
repository: a LeRobot checkpoint behind a small HTTP server on the bench PC or
the Spark). The adapter sends the robot's state and both camera images and
receives a chunk of joint targets, which it then executes through the same
guarded waypoint path every other motion uses. Keeping torch out of the
harness process keeps the realtime control loop and the policy's GPU work in
separate address spaces, and lets the policy host be any machine the bench
can reach.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import requests

POLICY_URL_ENV = "CAP_HARNESS_POLICY_URL"
DEFAULT_POLICY_URL = "http://127.0.0.1:8765"
POLICY_TIMEOUT_S = 10.0


class PolicyClient(Protocol):
    """What the adapter needs from a policy host."""

    def act(self, request: Mapping[str, Any]) -> list[list[float]]:
        """Return an action chunk, rows of joint targets (7) plus a gripper width."""

    def reset(self, policy: str) -> None:
        """Clear any per-episode state (action queues, ensembles) on the host."""


def encode_image_png(rgb: np.ndarray) -> str:
    """An RGB image as base64 PNG: lossless, so the policy sees what the camera saw."""
    from PIL import Image

    array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class HttpPolicyClient:
    """A policy host reached over HTTP; the URL comes from the environment."""

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout_s: float = POLICY_TIMEOUT_S,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.url = (url or os.environ.get(POLICY_URL_ENV) or DEFAULT_POLICY_URL).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._post = post or requests.post

    def act(self, request: Mapping[str, Any]) -> list[list[float]]:
        response = self._post(self.url + "/act", json=dict(request), timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"policy host returned no actions: {str(payload)[:200]}")
        return actions

    def reset(self, policy: str) -> None:
        response = self._post(self.url + "/reset", json={"policy": policy}, timeout=self.timeout_s)
        response.raise_for_status()
