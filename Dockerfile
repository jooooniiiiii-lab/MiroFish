FROM python:3.11-slim

# Install Node.js 20.x LTS
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/requirements.txt ./backend/

# Install Node dependencies
RUN npm ci && npm ci --prefix frontend

# Install Python dependencies (using pip with --default-timeout for slow downloads)
RUN pip install --timeout 120 --no-cache-dir -r backend/requirements.txt

# Copy source code
COPY . .

# Build frontend for production
RUN cd frontend && npm run build

EXPOSE 7860

ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=7860
ENV HF_SPACE=true
ENV PYTHONUNBUFFERED=1

CMD ["python", "backend/run.py"]
