from fastapi import APIRouter, Body, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import PlanDateRequest, PlanDateResponse
from app.search_client import get_search_client
from app.services import handle_plan_date

router = APIRouter(tags=["plan-date"], dependencies=[Depends(verify_service_api_key)])


@router.post("/plan-date", response_model=PlanDateResponse)
async def plan_date(
    req: PlanDateRequest = Body(
        ...,
        openapi_examples={
            "minimal": {
                "summary": "Minimal payload",
                "description": "Only location is required.",
                "value": {"location": "Mumbai"},
            },
            "with_context": {
                "summary": "Payload with optional context",
                "description": "Backend can send budget, memory, calendar, and preferences when available.",
                "value": {
                    "location": "Mumbai",
                    "budget": 100,
                    "memory": {"partner_name": "Emily", "favorite_food": "Italian"},
                    "date_type": "anniversary",
                    "preferences": "quiet, outdoor seating",
                },
            },
        },
    )
) -> PlanDateResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_plan_date(req, llm, search)
