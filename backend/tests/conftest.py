import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="wisq-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp_dir}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp_dir}/uploads"
# Force the fake LLM/embedder regardless of a developer's .env (e.g.
# LLM_PROVIDER=openai/EMBEDDING_PROVIDER=openai for local manual testing) --
# the suite must stay hermetic: no network calls, no API costs, no flakiness
# from a real provider.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["SPARSE_EMBEDDING_PROVIDER"] = "fake"

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.models import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    # Ensures tables exist for tests that talk to the DB directly (versioning/
    # relationship unit tests) without going through the `client` fixture, whose
    # TestClient-triggered lifespan would otherwise be the only thing creating them.
    async def _create() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create())
    yield


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
