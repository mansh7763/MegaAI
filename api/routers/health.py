from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "mega-ai-api", "version": "1.0.0"}
