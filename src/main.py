from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os
import sys

# Import config, repository, and orchestrator
try:
    import config
    from db.repository import get_distinct_filters
    from services.orchestrator import get_recommendations_orchestrated
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    from db.repository import get_distinct_filters
    from services.orchestrator import get_recommendations_orchestrated

app = FastAPI(
    title="Zomato AI Restaurant Recommender API",
    description="Intelligent restaurant suggestion service using structured database filtering and Groq AI."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User Preference Payload schema (V2 Pydantic)
class PreferencePayload(BaseModel):
    location: str = Field(..., description="The neighborhood or city area.")
    cuisine: str | None = Field(None, description="Preferred cuisine (e.g. Italian, Chinese).")
    min_budget: int | None = Field(None, description="Minimum cost for two.")
    max_budget: int | None = Field(None, description="Maximum cost for two.")
    min_rating: float = Field(0.0, ge=0.0, le=5.0, description="Minimum star rating.")
    additional_preferences: str | None = Field(None, description="Custom notes (e.g. rooftop, quiet).")

@app.get("/api/filters")
def get_filters():
    """Returns a list of distinct locations and cuisines for dropdown inputs."""
    try:
        filters = get_distinct_filters()
        return filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database filter lookup error: {str(e)}")

@app.post("/api/recommendations")
def get_recommendations(payload: PreferencePayload):
    """
    Orchestrated recommendation endpoint.
    Pre-filters database candidates and personalizes recommendations using Groq LLM.
    """
    try:
        response = get_recommendations_orchestrated(payload.model_dump())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration recommendation error: {str(e)}")

# Mount static files (will be populated in Phase 6)
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
