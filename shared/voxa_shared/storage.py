"""S3-compatible object storage (recordings). Works with AWS S3, MinIO, SeaweedFS, R2, etc."""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache
def _client(public: bool = False):
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("S3_PUBLIC_ENDPOINT") if public else os.environ.get("S3_ENDPOINT")
    return boto3.client(
        "s3", endpoint_url=endpoint or os.environ.get("S3_ENDPOINT"), region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "voxa"), aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "voxa-secret-key"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 3}))


def bucket() -> str:
    return os.environ.get("S3_BUCKET", "voxa-recordings")


def ensure_bucket() -> None:
    c = _client()
    try:
        c.head_bucket(Bucket=bucket())
    except Exception:  # noqa: BLE001
        c.create_bucket(Bucket=bucket())


def put_file(key: str, path: str, content_type: str = "audio/wav") -> None:
    _client().upload_file(path, bucket(), key, ExtraArgs={"ContentType": content_type, "ServerSideEncryption": "AES256"}
                          if os.environ.get("S3_SSE", "0") == "1" else {"ContentType": content_type})


def delete(key: str) -> None:
    _client().delete_object(Bucket=bucket(), Key=key)


def presign_get(key: str, expires: int = 300) -> str:
    """Short-lived link signed for the endpoint the *browser* can reach (S3_PUBLIC_ENDPOINT)."""
    return _client(public=True).generate_presigned_url("get_object", Params={"Bucket": bucket(), "Key": key}, ExpiresIn=expires)
