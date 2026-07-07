"""
Memory is owned entirely by the backend. This module only formats memory
for prompt injection and validates memory_update suggestions the AI proposes
back — it never reads or writes any persistent store itself.
"""
from typing import Any, Dict, List

from app.schemas import MemoryUpdate

# Keys the AI is allowed to suggest updates for. Anything outside this set
# still gets returned, but is flagged, since it's the backend's schema to own.
KNOWN_MEMORY_KEYS = {
    "name",
    "budget",
    "favorite_food",
    "favorite_activity",
    "relationship_status",
    "partner_name",
    "anniversary",
}


def sanitize_memory_updates(raw_updates: List[Dict[str, Any]]) -> List[MemoryUpdate]:
    cleaned: List[MemoryUpdate] = []
    for item in raw_updates or []:
        key = item.get("key")
        if not key or "value" not in item:
            continue
        cleaned.append(MemoryUpdate(key=key, value=item["value"]))
    return cleaned
