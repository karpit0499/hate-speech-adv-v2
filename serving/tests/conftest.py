import pytest

@pytest.fixture(scope="session")
def client():
    # Imported lazily and inside the fixture, so unit tests that don't request
    # `client` never pay the cost of loading the model.
    from fastapi.testclient import TestClient
    from app.main import app          # this line loads the model once
    return TestClient(app)