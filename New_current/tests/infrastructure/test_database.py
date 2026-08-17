"""Database model, migration, repository, and tenant-isolation tests."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic.config import Config
from alembic.script import ScriptDirectory
from cryptography.fernet import Fernet
from fakeredis.aioredis import FakeRedis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import ProcessingStage, Report, User
from app.db.repositories.sqlalchemy_repositories import ReportRepositoryImpl
from app.infrastructure.cache import CacheKeyBuilder, RedisStageCache
from app.infrastructure.schemas import JobCreateRequest
from app.infrastructure.service import InfrastructureService


class _Queue:
    def __init__(self) -> None:
        self.enqueued = 0
        self.revoked: list[str] = []

    async def enqueue(self, job_id, request_id, stages):
        del job_id, request_id, stages
        self.enqueued += 1
        return "celery-task-1"

    async def revoke(self, task_id):
        self.revoked.append(task_id)


def test_schema_contains_authoritative_and_additive_tables() -> None:
    expected = {
        "users",
        "reports",
        "report_processing",
        "medical_entities",
        "simplifications",
        "model_outputs",
        "feedback",
        "voice_profiles",
        "voice_generations",
        "supported_dialects",
        "user_preferences",
        "entity_links",
        "embedding_records",
        "translations",
        "processing_jobs",
        "audit_logs",
        "model_registry",
    }
    assert set(Base.metadata.tables) == expected
    for table in Base.metadata.tables.values():
        assert "created_at" in table.c
        assert "updated_at" in table.c
        assert "version" in table.c
        assert any(column.primary_key for column in table.c)
    assert "content_sha256" in Base.metadata.tables["reports"].c
    assert "translated_text" in Base.metadata.tables["translations"].c


def test_alembic_has_one_reviewable_head() -> None:
    root = Path(__file__).resolve().parents[2]
    configuration = Config(root / "alembic.ini")
    configuration.set_main_option("script_location", str(root / "migrations"))
    scripts = ScriptDirectory.from_config(configuration)
    assert scripts.get_heads() == ["0001_initial_schema"]


def test_report_repository_enforces_tenant_scope_and_soft_delete() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        owner_id = uuid4()
        other_owner_id = uuid4()
        async with sessions() as session:
            session.add_all(
                [
                    User(user_id=owner_id, full_name="Synthetic Owner", email="a@example.test"),
                    User(
                        user_id=other_owner_id,
                        full_name="Other Synthetic Owner",
                        email="b@example.test",
                    ),
                ]
            )
            await session.flush()
            repository = ReportRepositoryImpl(session)
            report = await repository.create(
                user_id=owner_id,
                original_filename="deidentified.pdf",
                content_sha256="a" * 64,
                storage_url="s3://encrypted/reports/synthetic",
                file_type="pdf",
                language="en",
                upload_time=datetime.now(UTC),
                status="uploaded",
                page_count=1,
                file_size=1024,
            )
            await session.commit()
            assert await repository.get_by_id(report.report_id, owner_id=owner_id)
            assert await repository.get_by_id(report.report_id, owner_id=other_owner_id) is None
            assert await repository.soft_delete(report.report_id, owner_id=owner_id)
            assert report.version == 2
            await session.commit()
            assert await repository.get_by_id(report.report_id, owner_id=owner_id) is None
        await engine.dispose()

    asyncio.run(scenario())


def test_job_service_is_durable_idempotent_and_cancellable() -> None:
    async def scenario() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        owner_id = uuid4()
        report_id = uuid4()
        async with sessions() as session:
            session.add(
                User(user_id=owner_id, full_name="Synthetic Owner", email="jobs@example.test")
            )
            session.add(
                Report(
                    report_id=report_id,
                    user_id=owner_id,
                    original_filename="deidentified.pdf",
                    content_sha256="c" * 64,
                    storage_url="s3://encrypted/reports/job-synthetic",
                    file_type="pdf",
                    language="en",
                    upload_time=datetime.now(UTC),
                    status="uploaded",
                    page_count=1,
                    file_size=1024,
                )
            )
            await session.commit()
        redis = FakeRedis()
        cache = RedisStageCache(
            redis,
            key_builder=CacheKeyBuilder("ibhealth"),
            encryption_key=Fernet.generate_key().decode(),
            default_ttl_seconds=60,
            lock_ttl_seconds=30,
        )
        queue = _Queue()
        service = InfrastructureService(
            engine=engine,
            session_factory=sessions,
            cache=cache,
            queue=queue,
        )
        command = JobCreateRequest(
            report_id=report_id,
            stages=[ProcessingStage.OCR, ProcessingStage.NER],
            pipeline_version="pipeline-v1",
            configuration_version="configuration-v1",
            model_revision="model-revision-v1",
        )
        created = await service.create_job(command, owner_id=owner_id, request_id=uuid4())
        duplicate = await service.create_job(command, owner_id=owner_id, request_id=uuid4())
        assert created.job_id == duplicate.job_id
        assert duplicate.duplicate is True
        assert queue.enqueued == 1
        await service.delete_job(created.job_id, owner_id=owner_id)
        assert queue.revoked == ["celery-task-1"]
        await cache.close()
        await engine.dispose()

    asyncio.run(scenario())
