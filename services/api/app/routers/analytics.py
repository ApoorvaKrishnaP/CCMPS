import os
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.api.app.services.analytics_engine import stream_crowd_predictions

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Crowd Analytics"]
)

# RTSP credentials live server-side only — never sent to the browser
_RTSP_URL = "rtsp://admin:MeYyPa@10.96.188.165:8556"

# Define the incoming data contract (Pydantic payload validation)
class StreamRequest(BaseModel):
    video_path: str
    horizon_sec: int = 30

@router.get("/stream")
async def stream_video_analytics(
    request: Request,
    video_path: str = Query(..., description="Path to the target source video file"),
    horizon_sec: int = Query(30, description="Predictive risk forecasting horizon timeframe")
):
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
    

    # 3. Verify the file actually exists on the system disk boundary before executing OpenCV pipelines
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Specified video payload file footprint not found on source track: '{video_path}'"
        )

    # 4. Initialize our streaming background generator engine instance
    event_generator = stream_crowd_predictions(video_path, ml_models, horizon_sec=horizon_sec)

    # 5. Return an asynchronous streaming HTTP response pipe with the correct SSE text-stream media headers
    return StreamingResponse(event_generator, media_type="text/event-stream")


@router.get("/live")
async def stream_live_rtsp(
    request: Request,
    horizon_sec: int = Query(30, description="Forecasting horizon in seconds")
):
    """
    Connects to the configured RTSP camera and runs the identical analytics
    pipeline used for uploaded videos. Credentials stay server-side.
    """
    ml_models = request.app.state.ml_models
    if not ml_models:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    return StreamingResponse(
        stream_crowd_predictions(_RTSP_URL, ml_models, horizon_sec=horizon_sec),
        media_type="text/event-stream"
    )


from pydantic import BaseModel as _Base
from typing import Optional

class SimulateRequest(_Base):
    scenario: str  # "gate_closure" | "increased_inflow" | "evacuation"

ZONE_NAMES = ['Gate A', 'Gate B', 'Main Hall', 'Concourse']

BASE_SCORES = {'Gate A': 42, 'Gate B': 38, 'Main Hall': 61, 'Concourse': 70}

def _risk(score):
    if score < 30: return 'Safe'
    if score < 55: return 'Moderate'
    if score < 75: return 'High'
    return 'Critical'

@router.post("/simulate")
async def simulate_scenario(body: SimulateRequest):
    zones = []
    recommendations = []
    for name in ZONE_NAMES:
        base = BASE_SCORES[name]
        score = base
        if body.scenario == 'gate_closure':
            score = min(100, base + 35) if 'Gate' in name else min(100, base + 12)
        elif body.scenario == 'increased_inflow':
            score = min(100, base + 22)
        elif body.scenario == 'evacuation':
            score = max(0, base - 20) if 'Exit' in name else min(100, base + 28)
        zones.append({'name': name, 'score': score, 'risk': _risk(score)})

    if body.scenario == 'gate_closure':
        recommendations = [
            'Redirect crowd via North Wing corridor',
            'Open auxiliary Gate C immediately',
            'Deploy 3 crowd-control staff at Main Hall junction',
            'Broadcast PA announcement for alternate entry routes',
        ]
    elif body.scenario == 'increased_inflow':
        recommendations = [
            'Activate overflow holding zones in Concourse',
            'Increase staff at all gate entry points',
            'Monitor South Wing density every 30 seconds',
            'Pre-position medical response teams near exits',
        ]
    elif body.scenario == 'evacuation':
        recommendations = [
            'Open all Exit gates to maximum capacity',
            'Clear Main Hall — direct to Exit 1 and Exit 2',
            'Deploy staff along evacuation route corridors',
            'Issue PA evacuation announcement immediately',
        ]

    return {'scenario': body.scenario, 'zones': zones, 'recommendations': recommendations}
