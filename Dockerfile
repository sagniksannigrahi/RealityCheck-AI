# syntax=docker/dockerfile:1.7
# RealityCheck AI — API + worker image (CPU). Build args:
#   INSTALL_ML=true   install PyTorch (CPU wheels), Ultralytics and transformers (default)
#   INSTALL_ML=false  slim image that can only run in DEMO mode (fake models)

FROM node:22-bookworm-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ARG INSTALL_ML=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models/hf YOLO_CONFIG_DIR=/models/ultralytics \
    RC_STORAGE_ROOT=/data/storage RC_DETECTOR_WEIGHTS=/models/yolo11n.pt \
    RC_FRONTEND_DIST=/app/frontend/dist RC_DEMO_MANIFEST_DIR=/app/datasets/demo
# OpenCV runtime libraries (headless build still needs libglib/libgl stubs for some codecs)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_ML" = "true" ]; then EXTRAS="--extra ml"; else EXTRAS=""; fi; \
    uv sync --frozen --no-dev --no-install-project $EXTRAS
COPY backend ./backend
COPY alembic.ini ./
COPY scripts ./scripts
COPY datasets/demo ./datasets/demo
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_ML" = "true" ]; then EXTRAS="--extra ml"; else EXTRAS=""; fi; \
    uv sync --frozen --no-dev $EXTRAS
COPY --from=frontend /fe/dist ./frontend/dist
RUN useradd --create-home --uid 10001 rc && mkdir -p /data /models && chown -R rc:rc /data /models /app
USER rc
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
