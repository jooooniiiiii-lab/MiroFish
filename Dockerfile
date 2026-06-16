FROM python:3.11

# Install Node.js (>=18) + required tools
RUN apt-get update \
  && apt-get install -y --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (for caching)
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package-lock.json ./frontend/
COPY backend/pyproject.toml backend/uv.lock ./backend/

# Install dependencies (Node + Python)
RUN npm ci \
  && npm ci --prefix frontend \
  && cd backend && uv sync --frozen

# Copy source code
COPY . .

# Build frontend for production
RUN cd frontend && npm run build

# HF Spaces uses port 7860
EXPOSE 7860

# Start backend (Flask serves both API + built frontend on port 7860)
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=7860
ENV HF_SPACE=true

CMD ["uv", "run", "--directory", "backend", "python", "run.py"]
