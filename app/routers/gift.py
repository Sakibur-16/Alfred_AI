from fastapi import APIRouter, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import GiftRequest, GiftResponse
from app.search_client import get_search_client
from app.services import handle_gift

router = APIRouter(tags=["gift"], dependencies=[Depends(verify_service_api_key)])


@router.post("/gift", response_model=GiftResponse)
async def gift(req: GiftRequest) -> GiftResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_gift(req, llm, search)
