"""Encrypted Redis cache, deterministic identities, and single-flight locks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.infrastructure.errors import (
    InfrastructureConfigurationError,
    InfrastructureUnavailableError,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Every input that can change a deterministic pipeline result."""

    tenant_id: UUID
    document_hash: str
    stage: str
    pipeline_version: str
    model_revision: str
    configuration_version: str
    prompt_or_rule_version: str
    schema_version: str

    def validate(self) -> None:
        if not _SHA256.fullmatch(self.document_hash):
            raise ValueError("document_hash must be a lowercase SHA-256 digest")
        for name in (
            "stage",
            "pipeline_version",
            "model_revision",
            "configuration_version",
            "prompt_or_rule_version",
            "schema_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")


class CacheKeyBuilder:
    """Build tenant-isolated, version-sensitive Redis keys."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix.strip()
        if not self._prefix:
            raise ValueError("cache prefix must not be blank")

    def build(self, identity: CacheIdentity) -> str:
        identity.validate()
        version_payload = {
            "configuration": identity.configuration_version,
            "model": identity.model_revision,
            "pipeline": identity.pipeline_version,
            "prompt_or_rule": identity.prompt_or_rule_version,
            "schema": identity.schema_version,
        }
        version_digest = hashlib.sha256(
            json.dumps(version_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return (
            f"{self._prefix}:cache:v1:{identity.tenant_id}:{identity.stage}:"
            f"{identity.document_hash}:{version_digest}"
        )

    def lock_key(self, cache_key: str) -> str:
        return f"{cache_key}:lock"


class RedisStageCache:
    """Minimal encrypted JSON cache suitable for clinical stage outputs."""

    def __init__(
        self,
        redis: Redis,
        *,
        key_builder: CacheKeyBuilder,
        encryption_key: str,
        default_ttl_seconds: int,
        lock_ttl_seconds: int,
    ) -> None:
        if default_ttl_seconds <= 0 or lock_ttl_seconds <= 0:
            raise InfrastructureConfigurationError("Redis TTL values must be positive.")
        try:
            self._cipher = Fernet(encryption_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as exc:
            raise InfrastructureConfigurationError("Invalid Redis cache encryption key.") from exc
        self._redis = redis
        self._keys = key_builder
        self._default_ttl = default_ttl_seconds
        self._lock_ttl = lock_ttl_seconds

    async def get(self, identity: CacheIdentity) -> Any | None:
        key = self._keys.build(identity)
        try:
            payload = await self._redis.get(key)
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis cache is unavailable.") from exc
        if payload is None:
            return None
        try:
            plaintext = self._cipher.decrypt(payload)
            return json.loads(plaintext)
        except (InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InfrastructureUnavailableError(
                "Redis cache payload integrity check failed."
            ) from exc

    async def set(
        self, identity: CacheIdentity, value: Any, *, ttl_seconds: int | None = None
    ) -> str:
        key = self._keys.build(identity)
        ttl = ttl_seconds or self._default_ttl
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        plaintext = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        encrypted = self._cipher.encrypt(plaintext)
        try:
            await self._redis.set(key, encrypted, ex=ttl)
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis cache is unavailable.") from exc
        return key

    async def delete(self, identity: CacheIdentity) -> bool:
        try:
            return bool(await self._redis.delete(self._keys.build(identity)))
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis cache is unavailable.") from exc

    async def metadata(self, identity: CacheIdentity) -> dict[str, Any]:
        key = self._keys.build(identity)
        try:
            ttl = await self._redis.ttl(key)
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis cache is unavailable.") from exc
        return {"exists": ttl >= 0, "ttl_seconds": ttl if ttl >= 0 else None, "key": key}

    async def acquire_lock(self, identity: CacheIdentity, token: str) -> bool:
        key = self._keys.lock_key(self._keys.build(identity))
        try:
            return bool(await self._redis.set(key, token, nx=True, ex=self._lock_ttl))
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis coordination is unavailable.") from exc

    async def release_lock(self, identity: CacheIdentity, token: str) -> bool:
        key = self._keys.lock_key(self._keys.build(identity))
        try:
            result = await self._redis.eval(_RELEASE_LOCK_SCRIPT, 1, key, token)
        except RedisError as exc:
            raise InfrastructureUnavailableError("Redis coordination is unavailable.") from exc
        return bool(result)

    async def health(self) -> tuple[bool, str]:
        try:
            healthy = bool(await self._redis.ping())
        except RedisError:
            return False, "Redis is unreachable."
        return healthy, "Redis is reachable." if healthy else "Redis ping failed."

    async def statistics(self) -> dict[str, int | None]:
        try:
            info = await self._redis.info("stats")
        except RedisError:
            return {"hits": None, "misses": None, "keys": None}
        try:
            keys = sum(
                int(section.get("keys", 0))
                for name, section in (await self._redis.info("keyspace")).items()
                if name.startswith("db") and isinstance(section, dict)
            )
        except RedisError:
            keys = None
        return {
            "hits": int(info.get("keyspace_hits", 0)),
            "misses": int(info.get("keyspace_misses", 0)),
            "keys": keys,
        }

    async def close(self) -> None:
        await self._redis.aclose()
