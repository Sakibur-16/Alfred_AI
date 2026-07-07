import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.llm_client import BaseLLMClient
from app.search_client import SerpAPIClient

API_KEY = "test-key"


@pytest.fixture(autouse=True)
def _configure_settings(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", API_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("SERPAPI_KEY", "")  # default: no live search in tests
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLMClient(BaseLLMClient):
    """Returns canned JSON responses keyed by a substring match on the prompt,
    so each test can control exactly what "the model" says without any network call."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        # responses is a queue consumed in order; simplest possible fake.
        self._queue: List[Dict[str, Any]] = list(responses or [])
        self.calls: List[str] = []

    def queue(self, response: Dict[str, Any]) -> None:
        self._queue.append(response)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        if not self._queue:
            return json.dumps({"reply": "ok", "confidence": 0.9})
        return json.dumps(self._queue.pop(0))


class FakeSearchClient(SerpAPIClient):
    def __init__(self, places=None, flights=None, hotels=None):
        self._places = places if places is not None else []
        self._flights = flights if flights is not None else []
        self._hotels = hotels if hotels is not None else []

    async def search_places(self, query: str, location: str):
        return self._places

    async def search_flights(self, origin: str, destination: str, start_date: str, end_date: str):
        return self._flights

    async def search_hotels(self, destination: str, start_date: str, end_date: str):
        return self._hotels


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def fake_search():
    return FakeSearchClient()


@pytest.fixture
def client(monkeypatch, fake_llm, fake_search):
    import app.llm_client as llm_module
    import app.search_client as search_module

    monkeypatch.setattr(llm_module, "get_llm_client", lambda: fake_llm)
    monkeypatch.setattr(search_module, "get_search_client", lambda: fake_search)
    # Routers imported get_llm_client/get_search_client by reference at import
    # time, so patch each router module's bound name too.
    for mod in (main_module.chat, main_module.recommend, main_module.plan_date, main_module.coach, main_module.gift, main_module.travel):
        if hasattr(mod, "get_llm_client"):
            monkeypatch.setattr(mod, "get_llm_client", lambda: fake_llm)
        if hasattr(mod, "get_search_client"):
            monkeypatch.setattr(mod, "get_search_client", lambda: fake_search)

    return TestClient(main_module.app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY}
