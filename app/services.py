"""
Business logic for every endpoint. Routers stay thin; all prompt assembly,
search orchestration, and response construction lives here so it's easy to
unit test without spinning up FastAPI.
"""
import logging
from typing import Any, Dict, List, Optional

from app.intent import detect_intent
from app.llm_client import BaseLLMClient, LLMError
from app.memory import sanitize_memory_updates
from app.prompts import (
    ALFRED_PERSONA,
    CHAT_RESPONSE_PROMPT,
    COACH_PROMPT,
    GIFT_PROMPT,
    NO_SEARCH_RESULTS_PROMPT,
    PLAN_DATE_PROMPT,
    RANK_RECOMMENDATIONS_PROMPT,
    TRAVEL_PROMPT,
    format_calendar_block,
    format_location_block,
    format_memory_block,
)
from app.schemas import (
    ActionRequest,
    ChatRequest,
    ChatResponse,
    CoachRequest,
    CoachResponse,
    GiftRequest,
    GiftResponse,
    Intent,
    PlanDateRequest,
    PlanDateResponse,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    TimelineStep,
    TravelRequest,
    TravelResponse,
    Location,
    UserMemory,
)
from app.search_client import SearchError, SerpAPIClient

logger = logging.getLogger("alfred.services")


def _history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-8:])


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------

async def handle_chat(req: ChatRequest, llm: BaseLLMClient) -> ChatResponse:
    memory = req.memory or UserMemory()
    calendar = req.calendar or []
    location = req.location or Location()
    history = req.conversation_history or []

    intent, intent_confidence = await detect_intent(llm, req.message, history)

    prompt = CHAT_RESPONSE_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(memory),
        calendar_block=format_calendar_block(calendar),
        location_block=format_location_block(location),
        budget=req.budget if req.budget is not None else "not specified",
        intent=intent.value,
        history=_history_text(history),
        message=req.message,
        partner_name=(req.memory.partner_name if req.memory else None) or "your partner",
        favorite_food=(req.memory.favorite_food if req.memory else None) or "their favorite food",
    )

    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError as exc:
        logger.error("Chat completion failed: %s", exc)
        return ChatResponse(
            reply="I'm having trouble thinking that through right now — could you try again in a moment?",
            intent=intent,
            confidence=0.3,
            actions=[],
            recommendations=[],
            memory_updates=[],
        )

    actions = [ActionRequest(**a) for a in data.get("actions", []) if a.get("action")]
    memory_updates = sanitize_memory_updates(data.get("memory_updates", []))

    return ChatResponse(
        reply=data.get("reply", ""),
        intent=intent,
        confidence=float(data.get("confidence", intent_confidence)),
        actions=actions,
        recommendations=[],
        memory_updates=memory_updates,
    )


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

async def _fetch_place_results(search: SerpAPIClient, category: str, location: str, preferences: Optional[str]) -> List[Dict[str, Any]]:
    query = f"{category}" + (f" {preferences}" if preferences else "") + f" in {location}"
    try:
        return await search.search_places(query, location)
    except SearchError:
        return []


async def handle_recommend(req: RecommendRequest, llm: BaseLLMClient, search: SerpAPIClient) -> RecommendResponse:
    category = req.category.value

    if category == "hotel":
        raw_results = await search.search_places(f"hotels in {req.location}", req.location)
    else:
        raw_results = await _fetch_place_results(search, category, req.location, req.preferences)

    if not raw_results:
        prompt = NO_SEARCH_RESULTS_PROMPT.format(
            persona=ALFRED_PERSONA,
            category=category,
            location=req.location,
            budget=req.budget if req.budget is not None else "not specified",
            memory_block=format_memory_block(req.memory),
        )
        try:
            data = await llm.complete_json(ALFRED_PERSONA, prompt)
        except LLMError:
            data = {"reply": "I couldn't pull live results just now, and I don't want to guess at real venues — want me to try again shortly?", "confidence": 0.3}
        return RecommendResponse(recommendations=[], reply=data.get("reply", ""), confidence=float(data.get("confidence", 0.4)))

    prompt = RANK_RECOMMENDATIONS_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        budget=req.budget if req.budget is not None else "not specified",
        category=category,
        location=req.location,
        preferences=req.preferences or "none stated",
        search_results=raw_results,
    )
    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError:
        # Fall back to raw, unranked results rather than fabricating anything.
        recs = [Recommendation(**{**r, "category": category, "source": "serpapi"}) for r in raw_results[:5] if r.get("name")]
        return RecommendResponse(recommendations=recs, reply="Here's what I found nearby.", confidence=0.4)

    recs = [Recommendation(**{**r, "category": r.get("category") or category, "source": "serpapi"}) for r in data.get("recommendations", [])]
    return RecommendResponse(recommendations=recs, reply=data.get("reply", ""), confidence=float(data.get("confidence", 0.6)))


# ---------------------------------------------------------------------------
# /plan-date
# ---------------------------------------------------------------------------

async def handle_plan_date(req: PlanDateRequest, llm: BaseLLMClient, search: SerpAPIClient) -> PlanDateResponse:
    restaurants = await _fetch_place_results(search, "restaurant", req.location, req.preferences)
    activities = await _fetch_place_results(search, "activity", req.location, req.preferences)

    prompt = PLAN_DATE_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        calendar_block=format_calendar_block(req.calendar),
        budget=req.budget if req.budget is not None else "not specified",
        location=req.location,
        date_type=req.date_type or "date",
        preferences=req.preferences or "none stated",
        restaurant_results=restaurants or "none available",
        activity_results=activities or "none available",
    )

    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError as exc:
        logger.error("Plan-date completion failed: %s", exc)
        return PlanDateResponse(
            reply="I couldn't put together a full plan just now — mind trying again in a moment?",
            timeline=[],
            confidence=0.3,
        )

    timeline = [TimelineStep(**t) for t in data.get("timeline", [])]
    restaurant = Recommendation(**data["restaurant"]) if data.get("restaurant") and data["restaurant"].get("name") else None
    activity = Recommendation(**data["activity"]) if data.get("activity") and data["activity"].get("name") else None
    actions = [ActionRequest(**a) for a in data.get("actions", []) if a.get("action")]
    memory_updates = sanitize_memory_updates(data.get("memory_updates", []))

    return PlanDateResponse(
        reply=data.get("reply", ""),
        timeline=timeline,
        restaurant=restaurant,
        activity=activity,
        estimated_cost=data.get("estimated_cost"),
        travel_notes=data.get("travel_notes"),
        actions=actions,
        memory_updates=memory_updates,
        confidence=float(data.get("confidence", 0.6)),
    )


# ---------------------------------------------------------------------------
# /coach
# ---------------------------------------------------------------------------

async def handle_coach(req: CoachRequest, llm: BaseLLMClient) -> CoachResponse:
    prompt = COACH_PROMPT.format(
        persona=ALFRED_PERSONA,
        topic=req.topic.value,
        memory_block=format_memory_block(req.memory),
        history=_history_text(req.conversation_history),
        message=req.message or "",
    )
    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError:
        return CoachResponse(reply="I'm having trouble pulling that together right now — try again in a moment?", tips=[], confidence=0.3)

    return CoachResponse(
        reply=data.get("reply", ""),
        tips=list(data.get("tips", [])),
        confidence=float(data.get("confidence", 0.6)),
    )


# ---------------------------------------------------------------------------
# /gift
# ---------------------------------------------------------------------------

async def handle_gift(req: GiftRequest, llm: BaseLLMClient, search: SerpAPIClient) -> GiftResponse:
    location = req.location or "the user's area"
    favorite = req.memory.favorite_activity or req.memory.favorite_food or "gift"
    query = f"{favorite} gift ideas" + (f" for {req.occasion}" if req.occasion else "")

    try:
        raw_results = await search.search_places(query, location) if req.location else []
    except SearchError:
        raw_results = []

    prompt = GIFT_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        occasion=req.occasion or "not specified",
        budget=req.budget if req.budget is not None else "not specified",
        location=location,
        search_results=raw_results or "none available",
    )
    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError:
        return GiftResponse(reply="I couldn't pull gift ideas together right now — want me to try again shortly?", recommendations=[], confidence=0.3)

    recs = [Recommendation(**{**r, "category": "gift", "source": "serpapi" if raw_results else "general_advice"}) for r in data.get("recommendations", [])]
    memory_updates = sanitize_memory_updates(data.get("memory_updates", []))

    return GiftResponse(
        reply=data.get("reply", ""),
        recommendations=recs,
        confidence=float(data.get("confidence", 0.6)),
        memory_updates=memory_updates,
    )


# ---------------------------------------------------------------------------
# /travel
# ---------------------------------------------------------------------------

async def handle_travel(req: TravelRequest, llm: BaseLLMClient, search: SerpAPIClient) -> TravelResponse:
    try:
        flights = await search.search_flights(req.origin, req.destination, req.start_date, req.end_date)
    except SearchError:
        flights = []
    try:
        hotels = await search.search_hotels(req.destination, req.start_date, req.end_date)
    except SearchError:
        hotels = []
    try:
        activities = await search.search_places(f"things to do in {req.destination}", req.destination)
    except SearchError:
        activities = []

    prompt = TRAVEL_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        origin=req.origin,
        destination=req.destination,
        start_date=req.start_date,
        end_date=req.end_date,
        budget=req.budget if req.budget is not None else "not specified",
        preferences=req.preferences or "none stated",
        flight_results=flights or "none available",
        hotel_results=hotels or "none available",
        activity_results=activities or "none available",
    )
    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError:
        return TravelResponse(reply="I couldn't put together the travel plan just now — want me to try again shortly?", confidence=0.3)

    actions = [ActionRequest(**a) for a in data.get("actions", []) if a.get("action")]

    return TravelResponse(
        reply=data.get("reply", ""),
        flights=[Recommendation(**{**f, "category": "flight"}) for f in data.get("flights", [])],
        hotels=[Recommendation(**{**h, "category": "hotel"}) for h in data.get("hotels", [])],
        activities=[Recommendation(**{**a, "category": "activity"}) for a in data.get("activities", [])],
        estimated_cost=data.get("estimated_cost"),
        actions=actions,
        confidence=float(data.get("confidence", 0.6)),
    )
