FROM python:3.11-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONUNBUFFERED=1
WORKDIR /usr/src/wyoming-nanowakeword

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY wyoming_nanowakeword ./wyoming_nanowakeword

RUN pip install --no-cache-dir .

# Bake the feature/VAD models into the image so the server starts offline and
# without a first-run download (NanoWakeWord otherwise fetches these lazily).
RUN python -c "from nanowakeword.interpreter.models._registry import models; \
    models.melspectrogram_onnx; models.embedding_model_onnx; models.silero_vad_onnx"

EXPOSE 10400

HEALTHCHECK --start-period=30s --interval=30s --timeout=5s --retries=3 \
    CMD echo '{ "type": "describe" }' \
        | nc -w 2 localhost 10400 \
        | grep -iq "nanowakeword" \
        || exit 1

CMD ["wyoming-nanowakeword", "--uri", "tcp://0.0.0.0:10400", "--model-dir", "/share/nanowakeword"]
