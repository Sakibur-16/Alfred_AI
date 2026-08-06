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
    assert body["options"] == []


def test_plan_date_embeds_recommendation_into_matching_timeline_step(client, auth_headers, fake_llm, fake_search):
    # The frontend should be able to render the whole plan by looping over
    # `timeline` alone — each step that IS the restaurant/activity pick should
    # carry that recommendation's full details (image, price, rating) inline,
    # rather than requiring a separate cross-reference against restaurant/activity.
    fake_search._places = [{"name": "Trattoria Roma", "rating": 4.6, "image_url": "https://example.com/roma.jpg"}]
    fake_llm.queue({
        "reply": "Here's a lovely evening plan.",
        "timeline": [
            {"time": "6:00 PM", "activity": "Dinner", "location": "Trattoria Roma", "notes": "Window table", "venue": "restaurant"},
            {"time": "8:00 PM", "activity": "Movie", "location": "Downtown Cinema", "notes": "", "venue": "activity"},
            {"time": "9:30 PM", "activity": "Head home", "location": None, "notes": "", "venue": None},
        ],
        "restaurant": {
            "name": "Trattoria Roma", "category": "restaurant", "rating": 4.6,
            "image_url": "https://example.com/roma.jpg", "reason": "Cozy spot",
        },
        "activity": {"name": "Downtown Cinema", "category": "activity", "reason": "Matches favorite activity"},
        "estimated_cost": 95,
        "confidence": 0.88,
    })

    resp = client.post("/plan-date", json={"location": "Indianapolis", "budget": 100}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()

    dinner_step = body["timeline"][0]
    movie_step = body["timeline"][1]
    home_step = body["timeline"][2]

    assert dinner_step["recommendation"]["name"] == "Trattoria Roma"
    assert dinner_step["recommendation"]["image_url"] == "https://example.com/roma.jpg"
    assert movie_step["recommendation"]["name"] == "Downtown Cinema"
    assert home_step["recommendation"] is None


def test_plan_date_recommendation_carries_image_url_and_details(client, auth_headers, fake_llm, fake_search):
    fake_search._places = [{"name": "Trattoria Roma", "rating": 4.6, "image_url": "https://example.com/roma.jpg"}]
    fake_llm.queue({
        "reply": "Here's a lovely evening plan.",
        "timeline": [{"time": "6:00 PM", "activity": "Dinner", "location": "Trattoria Roma"}],
        "restaurant": {
            "name": "Trattoria Roma", "category": "restaurant", "reason": "Cozy spot",
            "image_url": "https://example.com/roma.jpg",
            "details": [{"label": "Ambiance", "description": "Warm, dim lighting and quiet corners."}],
        },
        "activity": None,
        "estimated_cost": 60,
        "confidence": 0.85,
    })

    resp = client.post("/plan-date", json={"location": "Indianapolis"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["restaurant"]["image_url"] == "https://example.com/roma.jpg"
    assert body["restaurant"]["details"][0]["label"] == "Ambiance"


def test_plan_date_returns_browsable_options_when_num_options_greater_than_one(client, auth_headers, fake_llm, fake_search):
    fake_search._places = [{"name": "Trattoria Roma", "rating": 4.6}]
    fake_llm.queue({
        "reply": "Here are a few ideas to consider.",
        "options": [
            {"name": "Coffee & Nature Walk", "description": "A quiet morning stroll.", "estimated_cost": 30, "date_type": "relaxed"},
            {"name": "Italian Dinner Experience", "description": "A five-course meal.", "estimated_cost": 140, "date_type": "dining"},
            {"name": "Mountain Hiking Adventure", "description": "A scenic guided trek.", "estimated_cost": 110, "date_type": "adventurous"},
        ],
        "confidence": 0.8,
    })

    resp = client.post(
        "/plan-date",
        json={"location": "Indianapolis", "budget": 150, "num_options": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["options"]) == 3
    assert body["options"][0]["name"] == "Coffee & Nature Walk"
    assert body["timeline"] == []
    assert body["restaurant"] is None
