FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit==1.40.0

# Copy application code
COPY feature_repo/    ./feature_repo/
COPY serving/         ./serving/
COPY frontend/        ./frontend/
COPY models/          ./models/
COPY data/processed/  ./data/processed/
COPY params.yaml      ./params.yaml
COPY src/             ./src/

# Create reports directory
RUN mkdir -p reports

# Supervisor config to run both FastAPI and Streamlit
RUN echo '[supervisord]\n\
nodaemon=true\n\
\n\
[program:fastapi]\n\
command=uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 2\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
\n\
[program:streamlit]\n\
command=streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
directory=/app\n\
environment=API_URL="http://localhost:8000"\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n' > /etc/supervisor/conf.d/app.conf

EXPOSE 8000 8501

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
