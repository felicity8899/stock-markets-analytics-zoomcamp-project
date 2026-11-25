# Phase 2: containerised weekly run.
#
# Build (needs Docker; NOT tested in the dev VM, which has no Docker):
#   docker build -t stock-zoomcamp .
# Run (mount a volume so the data cache survives between runs):
#   docker run --rm -v stock-data:/app/data stock-zoomcamp \
#       --tickers 50 --years 5 --universe market-cap --force-refresh
#
# The image is large (~2 GB): tensorflow-cpu and xgboost (which pulls a
# ~200 MB NVIDIA wheel) dominate it. For a slim image without the LSTM,
# remove tensorflow-cpu from requirements.txt -- the pipeline skips the
# LSTM cleanly when tensorflow is absent.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/

# Data cache lives in a volume so weekly runs reuse downloads.
VOLUME ["/app/data"]

ENTRYPOINT ["python", "main.py"]
