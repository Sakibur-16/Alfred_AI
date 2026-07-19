"""
Business logic for every endpoint. Routers stay thin; all prompt assembly,
search orchestration, and response construction lives here so it's easy to
unit test without spinning up FastAPI.
"""
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.intent import DetectedIntent, detect_intent
from app.llm_client import BaseLLMClient, LLMError
from app.memory import sanitize_memory_updates
from app.session_store import SessionStore
from app.prompts import (
    AIRPORT_CODE_PROMPT,
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
    CoachTopic,
    GiftRequest,
    GiftResponse,
    Intent,
    PlanDateRequest,
    PlanDateResponse,
    Recommendation,
    RecommendCategory,
    RecommendRequest,
    RecommendResponse,
    TimelineStep,
    TravelRequest,
    TravelResponse,
    Location,
    UserMemory,
)
from app.exchange_client import ExchangeError, ExchangeRateClient
from app.search_client import SearchError, SerpAPIClient

logger = logging.getLogger("alfred.services")


def _history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-8:])


def _budget_text(budget: Optional[float], currency: str) -> str:
    if budget is None:
        return "not specified"
    return f"{budget} {currency}"


_CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "BDT": "৳"}


def _format_price(amount: float, currency: str) -> str:
    symbol = _CURRENCY_SYMBOLS.get(currency)
    formatted = f"{amount:,.2f}"
    return f"{symbol}{formatted}" if symbol else f"{formatted} {currency}"


async def _localize_product_prices(
    results: List[Dict[str, Any]], exchange: ExchangeRateClient, target_currency: str
) -> List[Dict[str, Any]]:
    """Google Shopping results always come back priced in USD (see search_products);
    convert each result's price into the currency the user actually asked for."""
    localized = []
    for r in results:
        price_usd = r.get("extracted_price_usd")
        clean = {k: v for k, v in r.items() if k != "extracted_price_usd"}
        if target_currency != "USD" and price_usd is not None:
            try:
                converted = await exchange.convert(price_usd, "USD", target_currency)
                clean["price_level"] = _format_price(converted, target_currency)
            except ExchangeError as exc:
                logger.warning("Price conversion to %s failed (%s); leaving USD price as-is", target_currency, exc)
        localized.append(clean)
    return localized


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------

def _merge_memory(base: UserMemory, override: Optional[UserMemory]) -> UserMemory:
    """Caller-sent memory wins field-by-field over stored session memory —
    the backend may know things (e.g. from its own profile DB) this service
    hasn't been told yet, so its data always takes precedence when present."""
    if override is None:
        return base
    override_fields = override.model_dump(exclude_none=True)
    if not override_fields:
        return base
    merged = base.model_dump()
    merged.update(override_fields)
    return UserMemory(**merged)


async def handle_chat(
    req: ChatRequest, llm: BaseLLMClient, search: SerpAPIClient, exchange: ExchangeRateClient,
    sessions: Optional[SessionStore] = None,
) -> ChatResponse:
    """Chat is the master endpoint: once intent is known, if the request already
    carries enough structured detail (e.g. a known location), route internally to
    the matching specialist flow and fold its result into one unified response —
    the caller doesn't need to orchestrate /recommend, /plan-date, /gift, /travel,
    or /coach separately.

    When req.session_id is set, prior turns, memory, location, and any other
    slots (travel dates, coach topic, etc.) mentioned in earlier turns are
    loaded from the session store and merged with whatever this request
    carries, so the backend doesn't have to resend full context every call."""
    session_id = req.session_id or (str(uuid.uuid4()) if sessions is not None else None)
    stored = sessions.get(session_id) if (sessions and req.session_id) else None
    stored_slots = stored.slots if stored else {}

    memory = _merge_memory(stored.memory if stored else UserMemory(), req.memory)
    calendar = req.calendar or []
    # Caller-sent location wins; otherwise fall back to whatever city was
    # remembered earlier in this session (e.g. mentioned two turns ago).
    location = req.location or (stored.location if stored else None) or Location()
    history = (stored.history if stored else []) + (req.conversation_history or [])

    detected = await detect_intent(llm, req.message, history)
    intent, intent_confidence = detected.intent, detected.confidence
    if detected.location_city and not location.city:
        location = Location(city=detected.location_city, country=location.country)
    if sessions and session_id and location.city:
        sessions.set_location(session_id, location.city)

    # Caller-sent values win over session-remembered ones, which win over
    # what the intent classifier just extracted from this message.
    def _slot(field_name: str, req_value: Optional[str], detected_value: Optional[str]) -> Optional[str]:
        return req_value or stored_slots.get(field_name) or detected_value

    # Budget-aware intents shouldn't search until the budget question has been
    # answered from *somewhere* — a number (this request, session, user memory,
    # or free text like "around 50 dollars"), OR an explicit "budget doesn't
    # matter" (no number to pass along, but the question is still answered).
    # Otherwise Alfred should ask, per the persona's "ask, don't guess" rule,
    # rather than silently searching unfiltered and never mentioning cost.
    budget = req.budget if req.budget is not None else (stored_slots.get("budget") or memory.budget or detected.budget_amount)
    budget_no_limit = bool(stored_slots.get("budget_no_limit")) or detected.budget_no_limit
    if sessions and session_id:
        if budget is not None:
            sessions.update_slots(session_id, {"budget": budget})
        if budget_no_limit:
            sessions.update_slots(session_id, {"budget_no_limit": True})
    have_budget = budget is not None or budget_no_limit

    if intent in (Intent.restaurant_search, Intent.activity_planning) and location.city and have_budget:
        category = RecommendCategory.restaurant if intent == Intent.restaurant_search else RecommendCategory.activity
        sub_req = RecommendRequest(
            category=category, location=location.city, budget=budget, currency=req.currency,
            memory=memory, preferences=req.message,
        )
        rec = await handle_recommend(sub_req, llm, search, history=history)
        return _finalize_chat(
            req, sessions, session_id, memory,
            reply=rec.reply, intent=intent, confidence=rec.confidence,
            recommendations=rec.recommendations,
        )

    if intent == Intent.date_planning and location.city and have_budget:
        sub_req = PlanDateRequest(
            location=location.city, budget=budget, currency=req.currency,
            memory=memory, calendar=calendar, preferences=req.message,
        )
        plan = await handle_plan_date(sub_req, llm, search, history=history)
        recs = [r for r in (plan.restaurant, plan.activity) if r]
        return _finalize_chat(
            req, sessions, session_id, memory,
            reply=plan.reply, intent=intent, confidence=plan.confidence,
            actions=plan.actions, recommendations=recs, memory_updates=plan.memory_updates,
            timeline=plan.timeline, estimated_cost=plan.estimated_cost,
        )

    if intent == Intent.gift_suggestions and have_budget:
        sub_req = GiftRequest(
            budget=budget, currency=req.currency, memory=memory, location=location.city,
            preferences=req.message,
        )
        gift = await handle_gift(sub_req, llm, search, exchange, history=history)
        return _finalize_chat(
            req, sessions, session_id, memory,
            reply=gift.reply, intent=intent, confidence=gift.confidence,
            recommendations=gift.recommendations, memory_updates=gift.memory_updates,
        )

    if intent == Intent.travel_planning:
        origin = _slot("travel_origin", req.origin, detected.travel_origin)
        destination = _slot("travel_destination", req.destination, detected.travel_destination)
        start_date = _slot("travel_start_date", req.start_date, detected.travel_start_date)
        end_date = _slot("travel_end_date", req.end_date, detected.travel_end_date)
        if sessions and session_id:
            sessions.update_slots(session_id, {
                "travel_origin": origin, "travel_destination": destination,
                "travel_start_date": start_date, "travel_end_date": end_date,
            })
        if origin and destination and start_date and end_date and have_budget:
            sub_req = TravelRequest(
                origin=origin, destination=destination, start_date=start_date, end_date=end_date,
                budget=budget, currency=req.currency, memory=memory, preferences=req.message,
            )
            travel = await handle_travel(sub_req, llm, search)
            recs = travel.flights + travel.hotels + travel.activities
            return _finalize_chat(
                req, sessions, session_id, memory,
                reply=travel.reply, intent=intent, confidence=travel.confidence,
                actions=travel.actions, recommendations=recs, estimated_cost=travel.estimated_cost,
            )
        # Missing a required slot (dates, budget) — fall through to the
        # generic chat branch below, which asks a follow-up rather than guessing.

    if intent == Intent.coaching:
        topic_str = _slot("coach_topic", None, detected.coach_topic) or "relationship_advice"
        try:
            topic = CoachTopic(topic_str)
        except ValueError:
            topic = CoachTopic.relationship_advice
        if sessions and session_id:
            sessions.update_slots(session_id, {"coach_topic": topic.value})
        sub_req = CoachRequest(topic=topic, message=req.message, memory=memory, conversation_history=history)
        coach = await handle_coach(sub_req, llm)
        return _finalize_chat(
            req, sessions, session_id, memory,
            reply=coach.reply, intent=intent, confidence=coach.confidence, tips=coach.tips,
        )

    prompt = CHAT_RESPONSE_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(memory),
        calendar_block=format_calendar_block(calendar),
        location_block=format_location_block(location),
        budget=_budget_text(budget, req.currency),
        intent=intent.value,
        history=_history_text(history),
        message=req.message,
        partner_name=memory.partner_name or "your partner",
        favorite_food=memory.favorite_food or "their favorite food",
    )

    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
    except LLMError as exc:
        logger.error("Chat completion failed: %s", exc)
        return _finalize_chat(
            req, sessions, session_id, memory,
            reply="I'm having trouble thinking that through right now — could you try again in a moment?",
            intent=intent, confidence=0.3,
        )

    actions = [ActionRequest(**a) for a in data.get("actions", []) if a.get("action")]
    memory_updates = sanitize_memory_updates(data.get("memory_updates", []))

    return _finalize_chat(
        req, sessions, session_id, memory,
        reply=data.get("reply", ""),
        intent=intent,
        confidence=float(data.get("confidence", intent_confidence)),
        actions=actions,
        memory_updates=memory_updates,
    )


def _finalize_chat(
    req: ChatRequest, sessions: Optional[SessionStore], session_id: Optional[str], memory: UserMemory,
    *, reply: str, intent: Intent, confidence: float,
    actions: Optional[List[ActionRequest]] = None,
    recommendations: Optional[List[Recommendation]] = None,
    memory_updates: Optional[List[Any]] = None,
    timeline: Optional[List[TimelineStep]] = None,
    estimated_cost: Optional[float] = None,
    tips: Optional[List[str]] = None,
) -> ChatResponse:
    """Persists the turn + any memory updates to the session (if one is active)
    and builds the ChatResponse. Centralized so every return path in handle_chat
    saves state the same way — easy to miss one if inlined at each call site."""
    memory_updates = memory_updates or []
    if sessions and session_id:
        sessions.append_turn(session_id, "user", req.message)
        sessions.append_turn(session_id, "assistant", reply)
        if memory_updates:
            memory = sessions.merge_memory(session_id, memory_updates)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        intent=intent,
        confidence=confidence,
        actions=actions or [],
        recommendations=recommendations or [],
        memory_updates=memory_updates,
        timeline=timeline or [],
        estimated_cost=estimated_cost,
        tips=tips or [],
    )


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

async def _fetch_place_results(search: SerpAPIClient, category: str, location: str, preferences: Optional[str], currency: Optional[str] = None) -> List[Dict[str, Any]]:
    query = f"{category}" + (f" {preferences}" if preferences else "") + f" in {location}"
    try:
        return await search.search_places(query, location, currency=currency)
    except SearchError:
        return []


async def handle_recommend(
    req: RecommendRequest, llm: BaseLLMClient, search: SerpAPIClient,
    history: Optional[List[Dict[str, str]]] = None,
) -> RecommendResponse:
    category = req.category.value

    if category == "hotel":
        raw_results = await search.search_places(f"hotels in {req.location}", req.location, currency=req.currency)
    else:
        raw_results = await _fetch_place_results(search, category, req.location, req.preferences, req.currency)

    if not raw_results:
        prompt = NO_SEARCH_RESULTS_PROMPT.format(
            persona=ALFRED_PERSONA,
            category=category,
            location=req.location,
            budget=_budget_text(req.budget, req.currency),
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
        budget=_budget_text(req.budget, req.currency),
        category=category,
        location=req.location,
        preferences=req.preferences or "none stated",
        search_results=raw_results,
        history=_history_text(history or []),
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

async def handle_plan_date(
    req: PlanDateRequest, llm: BaseLLMClient, search: SerpAPIClient,
    history: Optional[List[Dict[str, str]]] = None,
) -> PlanDateResponse:
    # Independent searches — run concurrently instead of paying for both round-trips serially.
    restaurants, activities = await asyncio.gather(
        _fetch_place_results(search, "restaurant", req.location, req.preferences, req.currency),
        _fetch_place_results(search, "activity", req.location, req.preferences, req.currency),
    )

    prompt = PLAN_DATE_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        calendar_block=format_calendar_block(req.calendar),
        budget=_budget_text(req.budget, req.currency),
        location=req.location,
        date_type=req.date_type or "date",
        preferences=req.preferences or "none stated",
        restaurant_results=restaurants or "none available",
        activity_results=activities or "none available",
        history=_history_text(history or []),
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

async def handle_gift(
    req: GiftRequest, llm: BaseLLMClient, search: SerpAPIClient, exchange: ExchangeRateClient,
    history: Optional[List[Dict[str, str]]] = None,
) -> GiftResponse:
    location = req.location or "the user's area"
    # Preferences (what the user actually asked for, e.g. "something more
    # meaningful") take priority over generic memory-derived interests — using
    # only memory meant every gift request with the same memory/occasion/budget
    # produced the identical search query, and therefore identical cached
    # results, no matter what the user actually said in this turn.
    favorite = req.preferences or req.memory.favorite_activity or req.memory.favorite_food or "unique"
    budget_hint = f" under {req.budget} {req.currency}" if req.budget is not None else ""
    query = f"{favorite} gift ideas" + (f" for {req.occasion}" if req.occasion else "") + budget_hint

    budget_usd: Optional[float] = None
    if req.budget is not None:
        try:
            budget_usd = await exchange.convert(req.budget, req.currency, "USD")
        except ExchangeError as exc:
            logger.warning("Currency conversion for gift budget failed (%s); searching without a price filter", exc)

    try:
        raw_results = await search.search_products(query, req.location)
    except SearchError:
        raw_results = []

    if budget_usd is not None:
        # Google's own price-range filter turned out to be unreliable (see
        # search_products) — filter on the real extracted price ourselves so
        # results actually respect the budget instead of just mentioning it.
        raw_results = [r for r in raw_results if r.get("extracted_price_usd") is not None and r["extracted_price_usd"] <= budget_usd]
        raw_results.sort(key=lambda r: r["extracted_price_usd"], reverse=True)
    raw_results = raw_results[:5]

    if raw_results:
        raw_results = await _localize_product_prices(raw_results, exchange, req.currency)

    prompt = GIFT_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        occasion=req.occasion or "not specified",
        budget=_budget_text(req.budget, req.currency),
        location=location,
        search_results=raw_results or "none available",
        history=_history_text(history or []),
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

async def _resolve_airport_code(llm: BaseLLMClient, place: str) -> Optional[str]:
    """Google Flights (via SerpAPI) requires a strict IATA 3-letter code, not a
    free-text city name — origin/destination in TravelRequest are plain city
    names, so ask the LLM to resolve the real airport code before searching,
    rather than passing the city straight through and always getting a 400."""
    try:
        data = await llm.complete_json(ALFRED_PERSONA, AIRPORT_CODE_PROMPT.format(place=place))
    except LLMError:
        return None
    code = data.get("code")
    if isinstance(code, str) and len(code) == 3 and code.isalpha():
        return code.upper()
    return None


async def _search_hotels_safe(search: SerpAPIClient, req: TravelRequest) -> List[Dict[str, Any]]:
    try:
        return await search.search_hotels(req.destination, req.start_date, req.end_date, currency=req.currency)
    except SearchError:
        return []


async def _search_activities_safe(search: SerpAPIClient, req: TravelRequest) -> List[Dict[str, Any]]:
    try:
        return await search.search_places(f"things to do in {req.destination}", req.destination, currency=req.currency)
    except SearchError:
        return []


async def handle_travel(req: TravelRequest, llm: BaseLLMClient, search: SerpAPIClient) -> TravelResponse:
    # Airport-code resolution (2 LLM calls) and hotel/activity search (2 SerpAPI
    # calls) are all independent of each other — run concurrently rather than
    # paying for four sequential round-trips.
    origin_code, destination_code, hotels, activities = await asyncio.gather(
        _resolve_airport_code(llm, req.origin),
        _resolve_airport_code(llm, req.destination),
        _search_hotels_safe(search, req),
        _search_activities_safe(search, req),
    )

    flights: List[Dict[str, Any]] = []
    if origin_code and destination_code:
        try:
            flights = await search.search_flights(origin_code, destination_code, req.start_date, req.end_date, currency=req.currency)
        except SearchError:
            flights = []
    else:
        logger.warning("Could not resolve airport codes for %r -> %r; skipping live flight search", req.origin, req.destination)

    prompt = TRAVEL_PROMPT.format(
        persona=ALFRED_PERSONA,
        memory_block=format_memory_block(req.memory),
        origin=req.origin,
        destination=req.destination,
        start_date=req.start_date,
        end_date=req.end_date,
        budget=_budget_text(req.budget, req.currency),
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
