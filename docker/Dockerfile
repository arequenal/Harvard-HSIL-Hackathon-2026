FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "matplotlib>=3.10.8" \
    "networkx>=3.4.2" \
    "pandas>=2.2.3" \
    "osmnx>=2.1.0" \
    "faster-whisper>=1.1.1" \
    "imageio-ffmpeg>=0.6.0" \
    "xgboost>=2.1.4" \
    "streamlit>=1.56.0" \
    "scikit-learn>=1.8.0" \
    "numpy>=2.2.0"

COPY . .

EXPOSE 8501 8502
