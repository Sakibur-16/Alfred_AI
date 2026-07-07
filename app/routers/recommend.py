from fastapi import APIRouter, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import RecommendRequest, RecommendResponse
from app.search_client import get_search_client
from app.services import handle_recommend

router = APIRouter(tags=["recommend"], dependencies=[Depends(verify_service_api_key)])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    llm = get_llm_client()
    search = get_search_client()
    return await handle_recommend(req, llm, search)
