"""In-memory per-client rate limiter (sliding minute window + daily budget).

Protects the shared, quota-limited LLM budget from a single noisy client.
For multi-replica deployments swap this for a Redis-backed limiter.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    retry_after_s: int = 0
    reason: str = ""


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, max_clients: int = 50_000) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.max_clients = max_clients
        self._minute: dict[str, deque[float]] = {}
        self._day: dict[str, tuple[int, int]] = {}  # client -> (day_number, count)
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> Decision:
        now = time.time() if now is None else now
        day = int(now // 86400)
        with self._lock:
            if len(self._minute) > self.max_clients:
                self._minute.clear()
                self._day.clear()
            window = self._minute.setdefault(client, deque())
            while window and window[0] <= now - 60:
                window.popleft()
            day_no, count = self._day.get(client, (day, 0))
            if day_no != day:
                day_no, count = day, 0
            if self.per_day and count >= self.per_day:
                return Decision(False, int((day + 1) * 86400 - now) + 1, "daily message limit reached")
            if self.per_minute and len(window) >= self.per_minute:
                return Decision(False, int(60 - (now - window[0])) + 1, "too many messages per minute")
            window.append(now)
            self._day[client] = (day_no, count + 1)
            return Decision(True)
