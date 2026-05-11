FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., git config, compiler tools if needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies and spacy model
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Run the FastAPI app (Hugging Face Spaces exposes port 7860 by default)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]