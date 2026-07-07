"""
Thin SerpAPI wrapper. The AI never fabricates recommendations — if this
returns [] or raises, callers must fall back to the "no search results"
prompt path rather than inventing venues.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

logger = logging.getLogger("alfred.search")


class SearchError(Exception):
    pass


class SerpAPIClient:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()

    @property
    def _enabled(self) -> bool:
        return bool(self._settings.serpapi_key)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params, "api_key": self._settings.serpapi_key}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(self._settings.serpapi_base_url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def search_places(self, query: str, location: str) -> List[Dict[str, Any]]:
        """Restaurants / activities / gifts-as-shops via Google Local results."""
        if not self._enabled:
            logger.warning("SERPAPI_KEY not set — skipping live search for %r", query)
            return []
        try:
            data = await self._get({
                "engine": "google_local",
                "q": query,
                "location": location,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("SerpAPI place search failed for %r: %s", query, exc)
            raise SearchError(str(exc)) from exc

        results = []
        for item in data.get("local_results", [])[:10]:
            results.append({
                "name": item.get("title"),
                "rating": item.get("rating"),
                "price_level": item.get("price"),
                "address": item.get("address"),
                "url": item.get("links", {}).get("website") if isinstance(item.get("links"), dict) else item.get("link"),
            })
        return results

    async def search_flights(self, origin: str, destination: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        if not self._enabled:
            logger.warning("SERPAPI_KEY not set — skipping live flight search")
            return []
        try:
            data = await self._get({
                "engine": "google_flights",
                "departure_id": origin,
                "arrival_id": destination,
                "outbound_date": start_date,
                "return_date": end_date,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("SerpAPI flight search failed: %s", exc)
            raise SearchError(str(exc)) from exc

        results = []
        for item in data.get("best_flights", []) + data.get("other_flights", []):
            results.append({
                "name": " / ".join(f.get("airline", "") for f in item.get("flights", [])) or "Flight option",
                "price_level": item.get("price"),
                "url": data.get("search_metadata", {}).get("google_flights_url"),
            })
        return results[:10]

    async def search_hotels(self, destination: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        if not self._enabled:
            logger.warning("SERPAPI_KEY not set — skipping live hotel search")
            return []
        try:
            data = await self._get({
                "engine": "google_hotels",
                "q": f"hotels in {destination}",
                "check_in_date": start_date,
                "check_out_date": end_date,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("SerpAPI hotel search failed: %s", exc)
            raise SearchError(str(exc)) from exc

        results = []
        for item in data.get("properties", [])[:10]:
            results.append({
                "name": item.get("name"),
                "rating": item.get("overall_rating"),
                "price_level": item.get("rate_per_night", {}).get("lowest") if isinstance(item.get("rate_per_night"), dict) else None,
                "url": item.get("link"),
            })
        return results


_search_singleton: Optional[SerpAPIClient] = None


def get_search_client() -> SerpAPIClient:
    global _search_singleton
    if _search_singleton is None:
        _search_singleton = SerpAPIClient()
    return _search_singleton


def reset_search_client_for_tests() -> None:
    global _search_singleton
    _search_singleton = None
