FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for LightGBM and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . /app

# Ensure model artifacts exist or train if missing
RUN python -m src.models.train

# Expose ports: 8000 for FastAPI REST service, 8501 for Streamlit UI
EXPOSE 8000 8501

# Default entrypoint starts FastAPI microservice
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
