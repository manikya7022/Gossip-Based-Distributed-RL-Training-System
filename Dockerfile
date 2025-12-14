# Multi-stage build for Gossip-RL
# Stage 1: C++ Build
FROM ubuntu:22.04 AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    liburing-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY CMakeLists.txt .
COPY cpp/ cpp/
COPY tests/cpp/ tests/cpp/

RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DUSE_URING=ON && \
    ninja

# Stage 2: Python Build
FROM python:3.10-slim AS python-builder

RUN pip install poetry==1.7.1
RUN poetry config virtualenvs.create false

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-dev --no-interaction --no-ansi

COPY python/ python/

# Stage 3: Runtime
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    liburing2 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash gossip && \
    mkdir -p /app /data /logs && \
    chown -R gossip:gossip /app /data /logs

WORKDIR /app

# Copy C++ libraries
COPY --from=cpp-builder /build/build/cpp/libgossip_rpc.so /usr/local/lib/
RUN ldconfig

# Copy Python packages
COPY --from=python-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-builder /app/python /app/python

# Copy configs and scripts
COPY configs/ configs/
COPY scripts/ scripts/

USER gossip

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import gossip_rl; print('healthy')" || exit 1

# Default command
ENTRYPOINT ["python3", "-m", "gossip_rl.agent"]
CMD ["--config", "/app/configs/agent.yaml"]
