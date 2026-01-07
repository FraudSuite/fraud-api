from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.dashboard_ws import router as dashboard_router
from routes.predict import router as predict_router
from routes.dashboard_ws import stream_listener
import asyncio


app = FastAPI()

stream_task = None



origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://frontend3-kappa.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(dashboard_router)
app.include_router(predict_router)


@app.on_event("startup")
async def start_stream():
    global stream_task
    stream_task = asyncio.create_task(stream_listener())





