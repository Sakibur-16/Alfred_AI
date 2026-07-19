import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.dependencies import verify_service_api_key
from app.exchange_client import get_exchange_client
from app.llm_client import get_llm_client
from app.schemas import ChatRequest, ChatResponse
from app.search_client import get_search_client
from app.services import handle_chat
from app.session_store import get_session_store

router = APIRouter(tags=["chat"], dependencies=[Depends(verify_service_api_key)])
logger = logging.getLogger("alfred.routers.chat")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest = Body(
        ...,
        openapi_examples={
            "minimal": {
                "summary": "Minimal payload",
                "description": "Only message is required.",
                "value": {
                    "message": "Plan a cozy date night for Friday"
                },
            },
            "with_context": {
                "summary": "Payload with optional context",
                "description": "Backend can send memory, location, and history when available.",
                "value": {
                    "message": "Plan a cozy date night for Friday",
                    "conversation_id": "conv-123",
                    "memory": {"partner_name": "Emily", "favorite_food": "Italian", "budget": 120},
                    "location": {"city": "Mumbai", "country": "India"},
                    "budget": 120,
                    "currency": "USD",
                    "conversation_history": [
                        {"role": "user", "content": "She likes quiet places."}
                    ],
                },
            },
            "with_session": {
                "summary": "Payload using session memory",
                "description": "Omit session_id on your first call — Alfred generates one and returns "
                "it in the response; reuse that exact returned value on later calls in the same "
                "conversation so Alfred recalls prior turns/memory server-side. Sessions live in this "
                "service's memory only: they reset whenever the service restarts, and this example "
                "value is a placeholder, not a real live session — copy the session_id from your own "
                "first response instead of reusing this one.",
                "value": {
                    "message": "What about somewhere quieter instead?",
                    "session_id": "REPLACE_WITH_THE_session_id_FROM_YOUR_FIRST_RESPONSE",
                },
            },
        },
    )
) -> ChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=422, detail="`message` must not be empty.")
    llm = get_llm_client()
    search = get_search_client()
    exchange = get_exchange_client()
    sessions = get_session_store()
    return await handle_chat(req, llm, search, exchange, sessions=sessions)
