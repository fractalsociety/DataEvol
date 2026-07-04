from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from dataevol.config import DataEvolConfig


def extract_token(authorization: str | None, x_dataevol_token: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return x_dataevol_token


def require_token(config: DataEvolConfig):
    def dependency(
        authorization: Annotated[str | None, Header()] = None,
        x_dataevol_token: Annotated[str | None, Header(alias="X-DataEvol-Token")] = None,
    ) -> None:
        supplied = extract_token(authorization, x_dataevol_token)
        if not supplied or not secrets.compare_digest(supplied, config.api_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid DataEvol API token.",
            )

    return dependency
