# TasteFinder: AI-Powered Restaurant Recommendation System

TasteFinder is an intelligent, AI-driven restaurant recommendation engine. It combines the speed and deterministic filtering of a traditional relational database with the reasoning and personalization capabilities of Large Language Models (LLMs) to provide highly tailored dining suggestions.

## 🚀 Features

- **Hyper-Personalized Recommendations:** Goes beyond basic filtering. The AI reads your custom preferences (e.g., "quiet ambiance", "good for dates", "vegan options") and finds the best match.
- **Cascading Query Relaxation:** Never returns an empty screen. If your strict criteria yield 0 results, the system intelligently relaxes the parameters (like slightly lowering the minimum rating) to guarantee a relevant fallback recommendation.
- **AI Rationale:** Every recommendation comes with a generated explanation detailing *why* the AI chose this specific restaurant for you.
- **Pre-filtering Engine:** Reduces LLM token costs and latency by executing strict SQL queries to narrow down candidates before passing them to the AI.
- **Modern UI:** A sleek, dark-themed, glassmorphic UI built with Vanilla web technologies for lightning-fast load times.

## 🛠️ Technology Stack

### Frontend
- **HTML5 & Vanilla JavaScript**: No heavy frameworks. Uses modular JS components and the Fetch API.
- **CSS3**: Custom properties, CSS Grid/Flexbox, and smooth micro-animations for a premium feel.

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI (Provides extremely fast async endpoints and automatic Swagger documentation)
- **Data Processing**: Pandas (for initial Hugging Face dataset processing and cleaning)

### Data Layer
- **Database**: SQLite (Embedded lightweight database for ultra-fast pre-filtering)
- **Dataset**: Hugging Face Zomato Restaurant Dataset

### Artificial Intelligence
- **LLM Provider**: Groq API 
- **Models**: Built to use cutting edge open-source models like `llama-3.3-70b-versatile` for blazing-fast inference speeds.

## ⚙️ Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Keerttik/Restaurant_Recommendation.git
   cd Restaurant_Recommendation
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   LLM_API_KEY=your_groq_api_key_here
   ```

4. **Start the Application:**
   ```bash
   uvicorn src.main:app --reload
   ```

5. **Open the App:**
   Navigate to `http://localhost:8000` in your browser. The API documentation is available at `http://localhost:8000/docs`.

## 🌐 Deployment (Railway)

This application is pre-configured for zero-config deployment on [Railway](https://railway.app/).

1. Connect your GitHub repository to Railway.
2. Railway will automatically detect the `requirements.txt` and `Procfile`.
3. Add your `LLM_API_KEY` in the Railway Variables tab.
4. Railway will build and serve the application automatically!

## 🏗️ Architecture

The app uses a 3-tier architecture:
1. **User requests** are sent from the Vanilla JS frontend to the FastAPI backend.
2. The **Pre-Filtering Engine** queries the local SQLite DB to find the top 10 closest matches based on strict metrics (budget, location, cuisine).
3. The **Prompt Builder** sends those 10 candidates along with the user's custom preferences to the **Groq LLM**.
4. The LLM returns a structured JSON response containing the final, personalized top recommendations, which are sent back to the user.
