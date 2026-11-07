# Reproducible CPU inference image. A release checkpoint must be explicitly supplied.
FROM python:3.11-slim-bookworm AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md LICENSE_POLICY.md ./
COPY src/ ./src/
COPY cpp/ ./cpp/
RUN python -m pip install --prefix=/install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3" \
    && python -m pip install --prefix=/install . \
    && g++ -O2 -std=c++17 -Wall -Wextra -Werror cpp/csi_features.cpp -o /build/csi_features

FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CGF_ENV=production CGF_MODEL_PATH=/opt/crossguard/models/model.pt
RUN groupadd -g 10001 crossguard && useradd --create-home --uid 10001 --gid 10001 crossguard
COPY --from=builder /install /usr/local
COPY --from=builder /build/csi_features /usr/local/bin/csi_features
WORKDIR /opt/crossguard
COPY artifacts/ /opt/crossguard/models/
RUN chown -R crossguard:crossguard /opt/crossguard
USER 10001:10001
EXPOSE 8000
# Readiness requires a valid model checkpoint; liveness is independent.
HEALTHCHECK --interval=30s --timeout=4s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=3)"
CMD ["uvicorn", "crossguard_fi.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
