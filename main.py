from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.dashboard_ws import router as dashboard_router
from .routes.predict import router as predict_router


app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
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







