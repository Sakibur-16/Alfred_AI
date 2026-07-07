def test_gift_recommendations_use_memory(client, auth_headers, fake_llm, fake_search):
    fake_search._places = []
    fake_llm.queue({
        "reply": "Since Emily loves Italian food, a cooking class together could be a lovely anniversary gift.",
        "recommendations": [
            {"name": "Italian Cooking Class", "category": "gift", "reason": "Matches favorite food"},
        ],
        "confidence": 0.8,
        "memory_updates": [],
    })

    resp = client.post(
        "/gift",
        json={
            "occasion": "anniversary",
            "budget": 150,
            "memory": {"partner_name": "Emily", "favorite_food": "Italian"},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Emily" in body["reply"]
    assert len(body["recommendations"]) == 1
