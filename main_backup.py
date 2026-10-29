
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import initialize_database, save_ticket, get_all_tickets

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ticket_classifier.joblib"
STATIC_DIR = BASE_DIR / "static"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Support Ticket Intelligence API",
    description="Classifies support tickets and predicts their priority.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TicketRequest(BaseModel):
    ticket_text: str = Field(min_length=5, max_length=2000)


def predict_priority(ticket_text: str) -> str:
    """
    Temporary baseline rules.
    We will replace these with a separately trained priority model.
    """
    text = ticket_text.lower()

    high_signals = [
        "charged twice",
        "money was deducted",
        "without permission",
        "account has been locked",
        "did not receive",
        "crashes",
        "not working",
        "failed but",
        "damaged",
        "cannot log in",
    ]

    low_signals = [
        "please explain",
        "how can i",
        "estimated delivery",
        "copy of my",
        "help configuring",
    ]

    if any(signal in text for signal in high_signals):
        return "High"

    if any(signal in text for signal in low_signals):
        return "Low"

    return "Medium"


@app.on_event("startup")
def startup():
    initialize_database()


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "running", "model_loaded": True}


@app.post("/api/predict")
def predict_ticket(request: TicketRequest):
    ticket_text = request.ticket_text.strip()

    if len(ticket_text) < 5:
        raise HTTPException(
            status_code=422,
            detail="Please enter a longer support ticket.",
        )

    category = str(model.predict([ticket_text])[0])
    priority = predict_priority(ticket_text)

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
