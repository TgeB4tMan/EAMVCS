from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def convert_g2p():
    return {"message": "G2P module ready"}
