# Stock Health Anomaly Report

## Description

This project demonstrates that an agentic AI architecture can turn a natural-language prompt into an anomaly-centered stock health report. Given a natural-language query (e.g., “Explain recent abnormal moves in Microsoft (MSFT), focusing on board changes, analyst upgrades/downgrades, and Fed shocks”), the system identifies periods of unusual price behavior relative to a benchmark (sector), retrieves relevant board events, analyst rating changes, macro factors, and news, and synthesizes them into one coherent narrative.

## Deployment Information

-   **Project Slug:** `stock-report`
-   **Deployment URL:** `https://[cloudfront-domain]/stock-report`
-   **Main File:** `app.py`

## Environment Variables Required

- `GEMINI_API_KEY`: Google Gemini API key
- `WRDS_USERNAME`: Username used for WRDS Library
- `WRDS_PASSWORD`: Password used for WRDS Library
 
## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## Docker Build
```
docker build --platform linux/amd64 -t stock-report:latest .
docker run \
  --env-file [.env file name ] \
  -p 8501:8501 stock-report:latest
```
