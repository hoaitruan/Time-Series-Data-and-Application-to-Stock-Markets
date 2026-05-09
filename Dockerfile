# ── Dockerfile ────────────────────────────────────────────────────────────────
# Build:   docker build -t stock-predictor .
# API:     docker run -d -p 5000:5000 --name stock-api stock-predictor
# Dash:    docker run -d -p 8501:8501 --name stock-dash stock-predictor streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
# Test:    curl http://localhost:5000/health

FROM python:3.12-slim

LABEL maintainer="DL4AI Project"
LABEL description="CNN-LSTM Stock Prediction – REST API + Streamlit Dashboard + Airflow Pipeline"

WORKDIR /app

# Install dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model artefacts
COPY nasdaq_model_task11.keras  ./
COPY vn_model_task21.keras      ./
COPY vn_model_task31_buy.keras  ./
COPY vn_model_task32_sell.keras ./

# Copy application code
COPY app.py        ./
COPY dashboard.py  ./
COPY pipeline.py   ./
COPY dags/         ./dags/

# Expose REST API and Streamlit ports
EXPOSE 5000 8501

# Liveness probe for REST API
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Default: start the REST API; override CMD to run dashboard or pipeline
CMD ["python", "app.py"]
