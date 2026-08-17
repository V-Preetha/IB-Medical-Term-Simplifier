"""Redis cache identity/encryption and Celery configuration tests."""

import asyncio
from uuid import uuid4

from cryptography.fernet import Fernet
from fakeredis.aioredis import FakeRedis

from app.infrastructure.cache import CacheIdentity, CacheKeyBuilder, RedisStageCache
from app.infrastructure.celery_app import create_celery_app
from app.infrastructure.config import InfrastructureSettings
from app.infrastructure.tasks import celery_app as registered_celery_app


def _identity(**changes) -> CacheIdentity:
    values = {
        "tenant_id": uuid4(),
        "document_hash": "b" * 64,
        "stage": "ocr",
        "pipeline_version": "pipeline-v1",
        "model_revision": "revision-v1",
        "configuration_version": "configuration-v1",
        "prompt_or_rule_version": "prompt-v1",
        "schema_version": "schema-v1",
    }
    values.update(changes)
    return CacheIdentity(**values)


def test_cache_keys_are_tenant_and_version_isolated() -> None:
    builder = CacheKeyBuilder("ibhealth")
    identity = _identity()
    first = builder.build(identity)
    assert first != builder.build(_identity(tenant_id=uuid4()))
    assert first != builder.build(_identity(tenant_id=identity.tenant_id, model_revision="r2"))
    assert identity.document_hash in first


def test_redis_cache_encrypts_payload_and_honors_delete() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        key = Fernet.generate_key().decode()
        identity = _identity()
        builder = CacheKeyBuilder("ibhealth")
        cache = RedisStageCache(
            redis,
            key_builder=builder,
            encryption_key=key,
            default_ttl_seconds=60,
            lock_ttl_seconds=30,
        )
        await cache.set(identity, {"result": "synthetic clinical output"})
        raw = await redis.get(builder.build(identity))
        assert raw is not None
        assert b"synthetic clinical output" not in raw
        assert await cache.get(identity) == {"result": "synthetic clinical output"}
        assert (await cache.metadata(identity))["ttl_seconds"] > 0
        assert await cache.delete(identity)
        assert await cache.get(identity) is None
        await cache.close()

    asyncio.run(scenario())


def test_celery_registers_durable_json_stage_tasks() -> None:
    settings = InfrastructureSettings(
        enabled=True,
        redis_url="redis://localhost:6379/0",
        redis_prefix="ibhealth",
        cache_ttl_seconds=3600,
        lock_ttl_seconds=300,
        cache_encryption_key=Fernet.generate_key().decode(),
        celery_broker_url="memory://",
        celery_result_backend="cache+memory://",
        celery_default_queue="ibhealth.cpu",
        celery_task_timeout_seconds=900,
        celery_max_retries=3,
        celery_retry_backoff_seconds=5,
        health_timeout_seconds=1,
    )
    application = create_celery_app(settings)
    assert application.conf.task_acks_late is True
    assert application.conf.task_reject_on_worker_lost is True
    assert application.conf.task_serializer == "json"
    assert set(application.conf.task_routes) == {
        "ibhealth.pipeline.ocr",
        "ibhealth.pipeline.ner",
        "ibhealth.pipeline.entity_linking",
        "ibhealth.pipeline.embeddings",
        "ibhealth.pipeline.simplification",
        "ibhealth.pipeline.translation",
        "ibhealth.infrastructure.recover_jobs",
    }
    for task_name in (
        "ibhealth.pipeline.ocr",
        "ibhealth.pipeline.ner",
        "ibhealth.pipeline.entity_linking",
        "ibhealth.pipeline.embeddings",
        "ibhealth.pipeline.simplification",
        "ibhealth.pipeline.translation",
        "ibhealth.infrastructure.recover_jobs",
    ):
        assert task_name in registered_celery_app.tasks
    assert "recover-durable-unsubmitted-jobs" in application.conf.beat_schedule
