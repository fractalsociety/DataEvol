from __future__ import annotations

import os

import uvicorn

from dataevol.specialist_server.app import create_server_app


def main() -> None:
    host = os.environ.get("SPECIALIST_SERVER_BIND", "127.0.0.1")
    port = int(os.environ.get("SPECIALIST_SERVER_PORT", "8767"))
    uvicorn.run(create_server_app(), host=host, port=port)


if __name__ == "__main__":
    main()
