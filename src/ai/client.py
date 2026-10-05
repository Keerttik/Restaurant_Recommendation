import os
import sys
import json
from pydantic import BaseModel, Field
from groq import Groq

# Import config
try:
    import config
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

# Define expected Pydantic schema validation structures for LLM recommendations
class AIRecommendationItem(BaseModel):
    rank: int = Field(..., description="The numerical ranking of this recommendation.")
    name: str = Field(..., description="The name of the restaurant.")
    cuisine: str = Field(..., description="The cuisines of the restaurant.")
    rating: float = Field(..., description="The rating score.")
    cost_for_two: int = Field(..., description="Average cost for two people.")
    currency: str = Field(..., description="The currency symbol (e.g. Rs.).")
    ai_explanation: str = Field(..., description="Personalized reasoning explaining why this restaurant matches the user's specific preferences.")

class AIRecommendationResponse(BaseModel):
    recommendations: list[AIRecommendationItem] = Field(..., description="List of top recommended restaurants.")

PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
]

def resolve_candidate_models(client: Groq, requested_model: str | None) -> list[str]:
    """
    Returns an ordered list of candidate models to try.
    Checks actual models available on the Groq account dynamically.
    """
    try:
        available = {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"Warning: Could not fetch Groq models list ({e})")
        available = set()

    candidates: list[str] = []

    # If user explicitly requested a model (that is not an OpenAI proprietary model like gpt-4o),
    # and it exists in available models (or available check failed), prioritize it.
    if requested_model and not requested_model.lower().startswith("gpt-4"):
        if not available or requested_model in available:
            candidates.append(requested_model)

    # Next add models from preferred list that are actually available
    for m in PREFERRED_GROQ_MODELS:
        if (not available or m in available) and m not in candidates:
            candidates.append(m)

    # If still empty or no preferred models available, add any text-generation models
    if available:
        for m in sorted(available):
            m_lower = m.lower()
            if not any(x in m_lower for x in ["whisper", "guard", "orpheus"]) and m not in candidates:
                candidates.append(m)

    if not candidates:
        candidates = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]

    return candidates

def get_groq_client():
    """Initializes and returns a Groq Client if API key is present."""
    if not config.LLM_API_KEY or config.LLM_API_KEY == "your_llm_api_key_here":
        raise ValueError("Groq LLM_API_KEY is missing. Please configure it in your .env file.")
    return Groq(api_key=config.LLM_API_KEY)

def generate_personalized_recommendations(user_prefs, candidate_restaurants):
    """
    Constructs the prompt, calls Groq API in JSON mode,
    and returns parsed recommendation objects.
    """
    if not candidate_restaurants:
        return {"recommendations": []}

    client = get_groq_client()
    models_to_try = resolve_candidate_models(client, config.LLM_MODEL)
    print(f"Candidate Groq models to try: {models_to_try}")

    # 1. Format candidate list
    candidates_formatted = []
    for idx, r in enumerate(candidate_restaurants):
        candidates_formatted.append(
            f"{idx + 1}. Name: {r['name']} | Location: {r['location']} | "
            f"Cuisines: {r['cuisines']} | Rating: {r['rating_number']} ({r['rating_text']}) | "
            f"Cost for two: {r['currency']}{r['average_cost_for_two']} | "
            f"Delivery: {'Yes' if r['has_online_delivery'] else 'No'} | "
            f"Table Booking: {'Yes' if r['has_table_booking'] else 'No'}"
        )
    candidates_text = "\n".join(candidates_formatted)

    min_b = user_prefs.get('min_budget')
    max_b = user_prefs.get('max_budget')
    budget_str = f"Rs. {min_b} to {f'Rs. {max_b}' if max_b and int(max_b) < 2000 else 'unlimited'}"

    # 2. Compile user prompt
    user_prompt = f"""
**User Specific Preferences**:
* Location: {user_prefs.get('location')}
* Preferred Cuisine: {user_prefs.get('cuisine', 'Any')}
* Budget Range: {budget_str}
* Minimum Rating: {user_prefs.get('min_rating', 'Any')}
* Additional Custom Preferences: "{user_prefs.get('additional_preferences', 'None')}"

**Candidate Restaurants (Pre-filtered from database)**:
{candidates_text}

Select up to 3 best matching unique restaurants, rank them by match quality, and write a personalized explanation for why each is chosen.
"""

    # 3. Compile system instructions enforcing JSON schema output
    json_schema_format = {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "name": {"type": "string"},
                        "cuisine": {"type": "string"},
                        "rating": {"type": "number"},
                        "cost_for_two": {"type": "integer"},
                        "currency": {"type": "string"},
                        "ai_explanation": {"type": "string"}
                    },
                    "required": ["rank", "name", "cuisine", "rating", "cost_for_two", "currency", "ai_explanation"]
                }
            }
        },
        "required": ["recommendations"]
    }

    system_instruction = f"""
You are Zomato's AI Restaurant Recommendation Assistant.
Your task is to select and rank the top 3 restaurants from the provided candidate list that best fit the user's preferences.
Do NOT suggest any restaurant that is not in the provided candidates list.
For each selected restaurant, write a concise, personalized explanation (2-3 sentences) directly addressing the user's explicit and implicit preferences.

You MUST return a JSON object adhering exactly to this JSON schema:
{json.dumps(json_schema_format, indent=2)}
"""

    # 4. Invoke Groq completion in JSON mode with fallback across candidate models
    last_error = None
    for model_name in models_to_try:
        try:
            print(f"Calling Groq completion using model '{model_name}'...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            # Load and parse output string
            raw_text = completion.choices[0].message.content
            result_json = json.loads(raw_text)

            # Validate schema using Pydantic model dump/load check
            AIRecommendationResponse(**result_json)

            return result_json

        except Exception as e:
            err_str = str(e).lower()
            last_error = e
            # If model is not found (404 / model_not_found) or unavailable, try next candidate
            if "not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                print(f"Model '{model_name}' unavailable on Groq: {e}. Trying next candidate model...")
                continue
            print(f"Error calling Groq API with model '{model_name}': {e}")
            raise e

    if last_error:
        print(f"All candidate Groq models failed. Last error: {last_error}")
        raise last_error
