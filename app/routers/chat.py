import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import ChatRequest, ChatResponse
from app.services import handle_chat

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
                    "conversation_history": [
                        {"role": "user", "content": "She likes quiet places."}
                    ],
                },
            },
        },
    )
) -> ChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=422, detail="`message` must not be empty.")
    llm = get_llm_client()
    return await handle_chat(req, llm)
