"""``python -m app`` - run the API server (container entrypoint)."""

from __future__ import annotations

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=30,
        log_config=None,  # structured logging is configured by the app
        access_log=False,
    )


if __name__ == "__main__":
    main()
