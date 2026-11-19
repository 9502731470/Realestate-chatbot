"""Outbound HTTPS: which certificate authorities to trust.

``TLS_TRUST=auto`` (default) verifies certificates against the operating
system's trust store on Windows and macOS, as browsers and pip do, and against
certifi's Mozilla bundle elsewhere (Linux, Docker). Company and campus networks
that inspect HTTPS install their own root certificate into Windows; Python's
bundled list doesn't contain it, so without this every request to an inspected
site fails with ``CERTIFICATE_VERIFY_FAILED``. Certificate verification is
never switched off.
"""

from __future__ import annotations

import logging
import ssl
import sys
from functools import lru_cache

log = logging.getLogger(__name__)

TLS_TRUST_MODES = ("auto", "system", "certifi")


def uses_system_store(mode: str) -> bool:
    return mode == "system" or (mode == "auto" and sys.platform in ("win32", "darwin"))


@lru_cache(maxsize=4)
def _system_context() -> ssl.SSLContext | None:
    try:
        import truststore
    except ImportError:  # pragma: no cover - truststore is a pinned dependency
        log.warning("truststore is not installed; verifying certificates with certifi instead")
        return None
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def tls_verify(mode: str = "auto") -> ssl.SSLContext | bool:
    """``verify=`` argument for httpx clients (True means certifi's bundle)."""
    if uses_system_store(mode):
        ctx = _system_context()
        if ctx is not None:
            return ctx
    return True


def describe_error(exc: BaseException, limit: int = 240) -> str:
    """Exception type and message, including the underlying cause, on one line."""
    parts = [f"{type(exc).__name__}: {exc}".rstrip(": ")]
    cause = exc.__cause__ or exc.__context__
    if cause is not None and str(cause) and str(cause) not in parts[0]:
        parts.append(f"{type(cause).__name__}: {cause}")
    return " <- ".join(parts)[:limit]
