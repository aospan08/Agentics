FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        g++ \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose Streamlit port
EXPOSE 8501

# IMPORTANT: Replace 'your-project-slug' with YOUR actual slug!
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=/stock-report"]