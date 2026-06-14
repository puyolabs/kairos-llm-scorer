# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings


def _present_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip()
    return x_api_key


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    accepted = settings.accepted_api_keys()
    if not accepted:
        return
    presented = _present_key(authorization, x_api_key)
    if presented is None or not any(secrets.compare_digest(presented, k) for k in accepted):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
