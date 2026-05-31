FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install system deps for audio processing + Python + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install faster-whisper with CUDA support
RUN pip3 install --no-cache-dir \
    faster-whisper>=1.0.0 \
    numpy>=1.24.0 \
    soundfile>=0.12.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6

COPY server.py .

EXPOSE 8000

CMD ["python3", "server.py"]
