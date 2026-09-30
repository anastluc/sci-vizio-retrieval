"""Serves LocalStorage signed URLs. Only mounted when STORAGE_BACKEND=local (development/tests)."""

from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from sci_vizio_saas.deps import get_services
from sci_vizio_saas.services import Services

router = APIRouter(prefix="/files", tags=["dev-storage"], include_in_schema=False)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _check(services: Services, method: str, key: str, expires: int, signature: str):
    storage = services.storage
    try:
        valid = storage.verify(method, key, expires, signature)
        path = storage.path_for(key)
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid or expired signature")
    return path


@router.put("/{key:path}")
async def put_file(key: str, request: Request, expires: int, signature: str,
                   services: Services = Depends(get_services)):
    _check(services, "PUT", key, expires, signature)
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Too large")
    services.storage.put(key, data, request.headers.get("content-type", "application/octet-stream"))
    return Response(status_code=status.HTTP_200_OK)


@router.get("/{key:path}")
def get_file(key: str, expires: int, signature: str, filename: Optional[str] = None,
             services: Services = Depends(get_services)):
    path = _check(services, "GET", key, expires, signature)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"} if filename else None
    return FileResponse(path, headers=headers)
