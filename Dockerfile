# ==========================================
# Stage 1: Build React Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and build static bundle
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend Runtime
# ==========================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /code

# Install system dependencies (build-essential, ffmpeg for voice input, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /code/requirements.txt

# Copy backend source code and datasets
COPY . /code

# Copy built frontend from Stage 1 into /code/frontend/dist
COPY --from=frontend-builder /app/frontend/dist /code/frontend/dist

# Create necessary runtime directories with write permissions
RUN mkdir -p /code/Data /code/usage-logs /data/.cache \
    && chmod -R 777 /code/Data /code/usage-logs /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
