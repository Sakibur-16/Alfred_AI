from fastapi import APIRouter, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import TravelRequest, TravelResponse
from app.search_client import get_search_client
from app.services import handle_travel

router = APIRouter(tags=["travel"], dependencies=[Depends(verify_service_api_key)])


@router.post("/travel", response_model=TravelResponse)
async def travel(req: TravelRequest) -> TravelResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_travel(req, llm, search)
