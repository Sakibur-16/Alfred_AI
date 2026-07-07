def test_chat_returns_structured_response(client, auth_headers, fake_llm):
    fake_llm.queue({"intent": "restaurant_search", "confidence": 0.85})
    fake_llm.queue({
        "reply": "Since Emily enjoys Italian food, here's a great pick nearby.",
        "confidence": 0.9,
        "actions": [{"action": "search_restaurants", "payload": {"location": "Indianapolis"}}],
        "memory_updates": [{"key": "favorite_food", "value": "Italian"}],
    })

    payload = {
        "message": "Find a nice Italian place for me and Emily",
        "conversation_id": "abc123",
        "memory": {"name": "Carl", "budget": 100, "partner_name": "Emily", "favorite_food": "Italian"},
        "budget": 100,
    }
    resp = client.post("/chat", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "restaurant_search"
    assert "Emily" in body["reply"]
    assert body["actions"][0]["action"] == "search_restaurants"
    assert body["memory_updates"][0]["key"] == "favorite_food"


def test_chat_rejects_empty_message(client, auth_headers):
    resp = client.post("/chat", json={"message": "  ", "conversation_id": "c1"}, headers=auth_headers)
    assert resp.status_code == 422


def test_chat_falls_back_gracefully_on_llm_error(client, auth_headers, fake_llm, monkeypatch):
    async def _boom(*args, **kwargs):
        from app.llm_client import LLMError
        raise LLMError("provider down")

    monkeypatch.setattr(fake_llm, "complete_json", _boom)

    resp = client.post(
        "/chat",
        json={"message": "hello", "conversation_id": "c1"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] <= 0.5


def test_chat_accepts_message_only_payload(client, auth_headers, fake_llm):
    fake_llm.queue({"intent": "small_talk", "confidence": 0.8})
    fake_llm.queue({"reply": "Hey there!", "confidence": 0.9, "actions": [], "memory_updates": []})

    resp = client.post(
        "/chat",
        json={"message": "hello"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "small_talk"
