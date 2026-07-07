from fastapi import APIRouter, Depends

from app.dependencies import verify_service_api_key
from app.llm_client import get_llm_client
from app.schemas import CoachRequest, CoachResponse
from app.services import handle_coach

router = APIRouter(tags=["coach"], dependencies=[Depends(verify_service_api_key)])


@router.post("/coach", response_model=CoachResponse)
async def coach(req: CoachRequest) -> CoachResponse:
    llm = get_llm_client()
    return await handle_coach(req, llm)
