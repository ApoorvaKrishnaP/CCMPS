import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.api.app.routers.analytics import router as analytics_router

# Global dictionary to hold pre-loaded model weights in RAM
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    12-Factor / Clean Architecture App Lifespan Handler.
    Pre-loads heavy ML models into memory ONCE during server startup.
    Cleans up resources when the server shuts down.
    """
    print("[STARTUP] Initializing system infrastructure...")
    
    # Define paths relative to this script layout
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "app", "models")
    
    try:
        # 1. Load YOLOv8 + Tracker weights
        print("[STARTUP] Pre-loading YOLOv8 Object Detection Weights...")
        import torch
        torch.set_num_threads(1)  # 1 intra-op thread per zone → zones run on separate CPU cores in parallel
        from ultralytics import YOLO
        yolo_path = os.path.join(models_dir, "yolov8n_best.pt")
        ml_models["yolo"] = YOLO(yolo_path)
        ml_models["yolo_path"] = yolo_path
        
        # 2. Load TensorFlow GRU Predictive Model
        print("[STARTUP] Pre-loading TensorFlow GRU Trend Predictor...")
        import tensorflow as tf
        ml_models["gru"] = tf.keras.models.load_model(os.path.join(models_dir, "gru_best.keras"))
        
        # 3. Load Scikit-Learn Scaler and Label Encoder
        print("[STARTUP] Pre-loading Data Transformers...")
        import pickle
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            ml_models["scaler"] = pickle.load(f)
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            ml_models["encoder"] = pickle.load(f)
            
        print("[STARTUP] All machine learning architectures successfully cached in RAM.")
        app.state.ml_models = ml_models
    except Exception as e:
        print(f"[CRITICAL ERR] Startup failure loading models: {e}")
        # Gracefully exit or handle initialization failures safely here
    
    yield
    # ---- Clean up phase during shutdown ----
    print("[SHUTDOWN] Flushing infrastructure caches and closing active pipes...")
    ml_models.clear()


# Initialize the core FastAPI Application framework instance
app = FastAPI(
    title="Crowd Analytics Predictive AI Engine",
    description="Asynchronous microservice analyzing crowd behavior patterns.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS so Node.js or a direct UI can reach this port securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to point strictly to your Node.js server domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(analytics_router)
@app.get("/health")
async def health_check():
    """Trivial health route for monitoring software readiness."""
    return {
        "status": "online", 
        "models_loaded": list(ml_models.keys())
    }