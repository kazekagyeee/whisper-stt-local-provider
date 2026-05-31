FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install system deps for audio processing + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install faster-whisper with CUDA support
RUN pip install --no-cache-dir \
    faster-whisper>=1.0.0 \
    numpy>=1.24.0 \
    soundfile>=0.12.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
