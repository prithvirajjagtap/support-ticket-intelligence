
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import initialize_database, save_ticket, get_all_tickets

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

CATEGORY_MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_classifier.joblib"

if not CATEGORY_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Category model missing. Run: python train_models.py"
    )

if not PRIORITY_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Priority model missing. Run: python train_models.py"
    )

category_model = joblib.load(CATEGORY_MODEL_PATH)
priority_model = joblib.load(PRIORITY_MODEL_PATH)

app = FastAPI(
    title="SupportIQ - Support Ticket Intelligence",
    description=(
        "Predicts customer support ticket categories and priorities "
        "using separately trained machine-learning models."
    ),
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TicketRequest(BaseModel):
    ticket_text: str = Field(min_length=5, max_length=2000)


@app.on_event("startup")
def startup():
    initialize_database()


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "running",
        "category_model_loaded": category_model is not None,
        "priority_model_loaded": priority_model is not None,
    }


@app.post("/api/predict")
def predict_ticket(request: TicketRequest):
    ticket_text = request.ticket_text.strip()

    if len(ticket_text) < 5:
        raise HTTPException(
            status_code=422,
            detail="Please enter a longer support ticket.",
        )

    # Use the same text format as the training dataset.
    model_input = "Subject: \nMessage: " + ticket_text

    category = str(category_model.predict([model_input])[0])
    priority = str(priority_model.predict([model_input])[0])

    ticket_id = save_ticket(ticket_text, category, priority)

    return {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "predicted_category": category,
        "predicted_priority": priority,
    }


@app.get("/api/tickets")
def list_tickets():
    return get_all_tickets()
