def test_plan_date_returns_full_timeline(client, auth_headers, fake_llm, fake_search):
    fake_search._places = [{"name": "Trattoria Roma", "rating": 4.6, "price_level": "$$"}]
    fake_llm.queue({
        "reply": "Here's a lovely evening plan for you and Emily.",
        "timeline": [
            {"time": "6:00 PM", "activity": "Dinner", "location": "Trattoria Roma", "notes": "Reserve a window table"},
            {"time": "8:00 PM", "activity": "Movie", "location": "Downtown Cinema", "notes": ""},
        ],
        "restaurant": {"name": "Trattoria Roma", "category": "restaurant", "reason": "Matches favorite food"},
        "activity": {"name": "Downtown Cinema", "category": "activity", "reason": "Matches favorite activity"},
        "estimated_cost": 95,
        "travel_notes": "Both venues are a 5 minute walk apart.",
        "actions": [{"action": "create_calendar_event", "payload": {"title": "Date night"}}],
        "memory_updates": [],
        "confidence": 0.88,
    })

    resp = client.post(
        "/plan-date",
        json={
            "location": "Indianapolis",
            "budget": 100,
            "memory": {"partner_name": "Emily", "favorite_food": "Italian", "favorite_activity": "Movies"},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["timeline"]) == 2
    assert body["restaurant"]["name"] == "Trattoria Roma"
    assert body["estimated_cost"] == 95
