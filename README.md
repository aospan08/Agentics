# Stock Health Anomaly Report

## Description

This project focuses on automating that workflow by generating anomaly-centered stock health reports. Given a natural-language query (e.g., “Explain recent abnormal moves in Microsoft (MSFT), focusing on board changes, analyst upgrades/downgrades, and Fed shocks”), the system identifies periods of unusual price behavior relative to a benchmark (sector), retrieves relevant board events, analyst rating changes, macro factors, and news, and synthesizes them into one coherent narrative.

## Deployment Information

-   **Project Slug:** `tbd`
-   **Deployment URL:** `tbd`
-   **Main File:** `app.py`

## Environment Variables Required

- `GEMINI_API_KEY`: Google Gemini API key
- `WRDS_USERNAME`: Username used for WRDS Library

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```
