from fastapi import APIRouter, Body, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import GiftRequest, GiftResponse
from app.search_client import get_search_client
from app.services import handle_gift

router = APIRouter(tags=["gift"], dependencies=[Depends(verify_service_api_key)])


@router.post("/gift", response_model=GiftResponse)
async def gift(
    req: GiftRequest = Body(
        ...,
        openapi_examples={
            "minimal": {
                "summary": "Minimal payload",
                "description": "Every field is optional — Alfred can still help with just an empty body.",
                "value": {"occasion": "birthday", "budget": 50},
            },
            "with_context": {
                "summary": "Payload with optional context",
                "description": "Backend can send memory and location when available.",
                "value": {
                    "occasion": "anniversary",
                    "budget": 150,
                    "location": "Mumbai",
                    "memory": {"partner_name": "Emily", "favorite_activity": "painting"},
                },
            },
        },
    )
) -> GiftResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_gift(req, llm, search)
