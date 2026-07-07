from app.memory import sanitize_memory_updates


def test_sanitize_memory_updates_drops_malformed_entries():
    raw = [
        {"key": "favorite_food", "value": "Japanese"},
        {"key": "", "value": "ignored"},
        {"value": "no key here"},
        {"key": "budget", "value": 120},
    ]
    result = sanitize_memory_updates(raw)
    assert len(result) == 2
    assert result[0].key == "favorite_food"
    assert result[1].value == 120


def test_sanitize_memory_updates_handles_empty_list():
    assert sanitize_memory_updates([]) == []
    assert sanitize_memory_updates(None) == []
