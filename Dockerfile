FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace model during build
RUN python -c "from insightface.app import FaceAnalysis; fa = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); fa.prepare(ctx_id=0)"

COPY . .

CMD gunicorn face_service:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
