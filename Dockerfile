FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY vla_data_adapter/ vla_data_adapter/

RUN pip install --no-cache-dir -e ".[all]"

VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["vla-adapter"]
CMD ["--help"]
