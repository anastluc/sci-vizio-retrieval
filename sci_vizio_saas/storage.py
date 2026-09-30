"""
Object storage for PDFs and figure images.

Browsers upload and download directly to/from storage through short-lived
signed URLs, so file bytes never pass through the API dynos.

* ``S3Storage`` — any S3-compatible service (Cloudflare R2, AWS S3, Google
  Cloud Storage in interoperability mode, MinIO).
* ``LocalStorage`` — files on disk, served by the API's ``/api/files`` route
  with HMAC-signed URLs. For development and tests only.
"""

import hashlib
import hmac
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import quote, urlencode


@dataclass
class PresignedUpload:
    url: str
    method: str = "PUT"
    headers: Dict[str, str] = field(default_factory=dict)


class ObjectStorage(ABC):
    @abstractmethod
    def presign_upload(self, key: str, content_type: str, expires: int = 900) -> PresignedUpload:
        """URL a browser can PUT the object to."""

    @abstractmethod
    def presign_download(self, key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
        """URL a browser can GET the object from."""

    @abstractmethod
    def size(self, key: str) -> Optional[int]:
        """Object size in bytes, or None if it doesn't exist."""

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def delete(self, keys: Iterable[str]) -> None:
        """Delete objects; missing keys are ignored."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Delete every object whose key starts with ``prefix``."""


class S3Storage(ObjectStorage):
    def __init__(self, bucket: str, endpoint_url: str = None, region: str = "auto",
                 access_key_id: str = None, secret_access_key: str = None, client=None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(signature_version="s3v4"),
            )
        self.bucket = bucket
        self.client = client

    def presign_upload(self, key, content_type, expires=900):
        url = self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires,
        )
        return PresignedUpload(url=url, headers={"Content-Type": content_type})

    def presign_download(self, key, expires=3600, filename=None):
        params = {"Bucket": self.bucket, "Key": key}
        if filename:
            params["ResponseContentDisposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
        return self.client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires)

    def size(self, key):
        from botocore.exceptions import ClientError

        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def get(self, key):
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def put(self, key, data, content_type):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def delete(self, keys):
        keys = list(keys)
        for start in range(0, len(keys), 1000):  # S3 batch limit
            batch = [{"Key": k} for k in keys[start:start + 1000]]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})

    def delete_prefix(self, prefix):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [obj["Key"] for obj in page.get("Contents", [])]
            if keys:
                self.delete(keys)


class LocalStorage(ObjectStorage):
    """Files under ``root``; URLs point at the API's ``/api/files/{key}`` route."""

    def __init__(self, root: str | Path, public_api_url: str, secret: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_api_url = public_api_url.rstrip("/")
        self._secret = secret.encode()

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Invalid storage key: {key!r}")
        return path

    def sign(self, method: str, key: str, expires_at: int) -> str:
        message = f"{method}\n{key}\n{expires_at}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def verify(self, method: str, key: str, expires_at: int, signature: str) -> bool:
        if expires_at < time.time():
            return False
        return hmac.compare_digest(self.sign(method, key, expires_at), signature)

    def _url(self, method: str, key: str, expires: int, extra: Dict[str, str] = None) -> str:
        expires_at = int(time.time()) + expires
        query = {"expires": expires_at, "signature": self.sign(method, key, expires_at), **(extra or {})}
        return f"{self.public_api_url}/api/files/{quote(key)}?{urlencode(query)}"

    def presign_upload(self, key, content_type, expires=900):
        return PresignedUpload(url=self._url("PUT", key, expires), headers={"Content-Type": content_type})

    def presign_download(self, key, expires=3600, filename=None):
        return self._url("GET", key, expires, {"filename": filename} if filename else None)

    def size(self, key):
        path = self.path_for(key)
        return path.stat().st_size if path.exists() else None

    def get(self, key):
        return self.path_for(key).read_bytes()

    def put(self, key, data, content_type):
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, keys):
        for key in keys:
            self.path_for(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix):
        if not prefix.endswith("/"):
            raise ValueError("LocalStorage only deletes whole directories; end the prefix with '/'")
        shutil.rmtree(self.path_for(prefix), ignore_errors=True)


def make_storage(settings) -> ObjectStorage:
    if settings.storage_backend == "s3":
        return S3Storage(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
        )
    return LocalStorage(settings.local_storage_dir, settings.public_api_url, settings.signing_secret)
