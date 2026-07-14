import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="wisq-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp_dir}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp_dir}/uploads"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client
