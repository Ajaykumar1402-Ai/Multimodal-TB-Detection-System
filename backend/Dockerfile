# Use official PyTorch GPU runtime image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set workspace directory
WORKDIR /workspace

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY . .

# Create models directory and expose port
RUN mkdir -p models
EXPOSE 8000

# Run FastAPI server
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
