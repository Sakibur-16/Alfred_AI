"""
All request/response contracts, matching the AI Developer Brief's
Output Format section exactly:

{
 "reply": "",
 "intent": "",
 "confidence": 0.98,
 "actions": [],
 "recommendations": [],
 "memory_updates": []
}
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    general_chat = "general_chat"
    restaurant_search = "restaurant_search"
    activity_planning = "activity_planning"
    date_planning = "date_planning"
    budget_planning = "budget_planning"
    travel_planning = "travel_planning"
    gift_suggestions = "gift_suggestions"
    coaching = "coaching"
    calendar_assistance = "calendar_assistance"
    anniversary_planning = "anniversary_planning"
    reminder_requests = "reminder_requests"
    small_talk = "small_talk"


class CalendarEvent(BaseModel):
    title: str
    date: str
    time: Optional[str] = None
    notes: Optional[str] = None


class UserMemory(BaseModel):
    """Mirrors the example memory object in the brief. Extra fields are allowed
    since the backend may evolve this shape independently of the AI layer."""
    name: Optional[str] = None
    budget: Optional[float] = None
    favorite_food: Optional[str] = None
    favorite_activity: Optional[str] = None
    relationship_status: Optional[str] = None
    partner_name: Optional[str] = None
    anniversary: Optional[str] = None

    model_config = {"extra": "allow"}


class Location(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ActionRequest(BaseModel):
    """An action the AI proposes; the backend decides whether/how to execute it."""
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    key: str
    value: Any


class Recommendation(BaseModel):
    name: str
    category: Optional[str] = None
    rating: Optional[float] = None
    price_level: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    reason: Optional[str] = None
    source: str = "serpapi"


class StructuredAIResponse(BaseModel):
    """The universal response envelope used by every endpoint."""
    reply: str
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    actions: List[ActionRequest] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    memory: Optional[UserMemory] = None
    location: Optional[Location] = None
    calendar: Optional[List[CalendarEvent]] = None
    budget: Optional[float] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    subscription_status: Optional[str] = None


class ChatResponse(StructuredAIResponse):
    pass


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

class RecommendCategory(str, Enum):
    restaurant = "restaurant"
    activity = "activity"
    gift = "gift"
    hotel = "hotel"


class RecommendRequest(BaseModel):
    category: RecommendCategory
    location: str
    budget: Optional[float] = None
    memory: UserMemory = Field(default_factory=UserMemory)
    preferences: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]
    reply: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# POST /plan-date
# ---------------------------------------------------------------------------

class PlanDateRequest(BaseModel):
    location: str
    budget: Optional[float] = None
    memory: UserMemory = Field(default_factory=UserMemory)
    calendar: List[CalendarEvent] = Field(default_factory=list)
    date_type: Optional[str] = None       # e.g. "first date", "anniversary"
    preferences: Optional[str] = None


class TimelineStep(BaseModel):
    time: str
    activity: str
    location: Optional[str] = None
    notes: Optional[str] = None


class PlanDateResponse(BaseModel):
    reply: str
    timeline: List[TimelineStep]
    restaurant: Optional[Recommendation] = None
    activity: Optional[Recommendation] = None
    estimated_cost: Optional[float] = None
    travel_notes: Optional[str] = None
    actions: List[ActionRequest] = Field(default_factory=list)
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# POST /coach
# ---------------------------------------------------------------------------

class CoachTopic(str, Enum):
    first_date = "first_date"
    second_date = "second_date"
    texting_advice = "texting_advice"
    relationship_advice = "relationship_advice"
    conversation_starters = "conversation_starters"


class CoachRequest(BaseModel):
    topic: CoachTopic
    message: Optional[str] = None
    memory: UserMemory = Field(default_factory=UserMemory)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class CoachResponse(BaseModel):
    reply: str
    tips: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# POST /gift
# ---------------------------------------------------------------------------

class GiftRequest(BaseModel):
    occasion: Optional[str] = None
    budget: Optional[float] = None
    memory: UserMemory = Field(default_factory=UserMemory)
    location: Optional[str] = None


class GiftResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    confidence: float = Field(ge=0.0, le=1.0)
    memory_updates: List[MemoryUpdate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /travel
# ---------------------------------------------------------------------------

class TravelRequest(BaseModel):
    origin: str
    destination: str
    start_date: str
    end_date: str
    budget: Optional[float] = None
    memory: UserMemory = Field(default_factory=UserMemory)
    preferences: Optional[str] = None


class TravelResponse(BaseModel):
    reply: str
    flights: List[Recommendation] = Field(default_factory=list)
    hotels: List[Recommendation] = Field(default_factory=list)
    activities: List[Recommendation] = Field(default_factory=list)
    estimated_cost: Optional[float] = None
    actions: List[ActionRequest] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    follow_up_question: Optional[str] = None
