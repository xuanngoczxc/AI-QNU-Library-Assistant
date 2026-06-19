FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache \
    TRANSFORMERS_CACHE=/data/.cache \
    SENTENCE_TRANSFORMERS_HOME=/data/.cache \
    PORT=7860

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /code/requirements.txt

# Tạo thư mục data persistent trước khi copy để tránh lỗi permission
RUN mkdir -p /data/pdfs /data/Data /data/.cache

# Copy code (không copy Data/ và pdfs/ vì sẽ mount riêng)
COPY . .

# Đảm bảo các thư mục tồn tại và writable
RUN chmod -R 777 /data /code

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]</newString>
