from fastapi import APIRouter, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import PlanDateRequest, PlanDateResponse
from app.search_client import get_search_client
from app.services import handle_plan_date

router = APIRouter(tags=["plan-date"], dependencies=[Depends(verify_service_api_key)])


@router.post("/plan-date", response_model=PlanDateResponse)
async def plan_date(req: PlanDateRequest) -> PlanDateResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_plan_date(req, llm, search)
