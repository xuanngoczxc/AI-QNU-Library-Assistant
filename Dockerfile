# ==========================================
# Stage 1: Build React Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Tận dụng cache npm để các lần build tiếp theo chỉ mất vài giây
COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit

# Copy mã nguồn frontend và build bundle tĩnh
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend Runtime
# ==========================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /code

# Chỉ cài đặt curl cho Healthcheck (loại bỏ gcc/build-essential và ffmpeg để tối ưu dung lượng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tận dụng pip cache mount để tránh tải lại wheels
COPY requirements.txt /code/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r /code/requirements.txt

# Copy toàn bộ mã nguồn backend và dữ liệu
COPY . /code

# Pre-generate file cache pickle ngay trong lúc build image (Container khởi động <0.5s, pass healthcheck ngay)
RUN python -c "from lib.search_engine import DocumentIndexer; DocumentIndexer.load_from_data('Data')"

# Copy static bundle từ Stage 1 sang backend
COPY --from=frontend-builder /app/frontend/dist /code/frontend/dist

# Phân quyền thư mục runtime
RUN mkdir -p /code/Data /code/usage-logs /data/.cache \
    && chmod -R 777 /code/Data /code/usage-logs /data

EXPOSE 8000

# Healthcheck start-period rút ngắn từ 40s xuống 15s vì đã có cache pickle sẵn
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
