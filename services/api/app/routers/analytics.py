import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.api.app.services.analytics_engine import stream_crowd_predictions

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Crowd Analytics"]
)

# Define the incoming data contract (Pydantic payload validation)
class StreamRequest(BaseModel):
    video_path: str
    horizon_sec: int = 30

@router.post("/stream")
async def stream_video_analytics(payload: StreamRequest, request: Request):
    """
    Exposes a real-time Server-Sent Events (SSE) data stream pipe 
    processing crowd predictive analytics for a specified video payload file.
    """
    # 1. Access the pre-loaded global model cache from our main app state
    ml_models = request.app.state.ml_models
    if not ml_models:
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model architecture cache is currently uninitialized."
        )

    # 2. Extract and validate parameters
    video_target = payload.video_path
    horizon = payload.horizon_sec

    # 3. Verify the file actually exists on the system disk boundary before executing OpenCV pipelines
    if not os.path.exists(video_target):
        raise HTTPException(
            status_code=404, 
            detail=f"Specified video payload file footprint not found on source track: '{video_target}'"
        )

    # 4. Initialize our streaming background generator engine instance
    event_generator = stream_crowd_predictions(video_target, ml_models, horizon_sec=horizon)

    # 5. Return an asynchronous streaming HTTP response pipe with the correct SSE text-stream media headers
    return StreamingResponse(
        event_generator, 
        media_type="text/event-stream"
    )