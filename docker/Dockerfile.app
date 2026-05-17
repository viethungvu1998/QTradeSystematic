FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QTS_ROOT=/var/lib/qts

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY qts ./qts
COPY configs ./configs
COPY lib/vectorbt.pro-main ./lib/vectorbt.pro-main

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[data,vn,research,execution,orchestration,tracking,tuning]"

RUN useradd --create-home --shell /bin/bash qts \
    && mkdir -p /var/lib/qts /mlflow/artifacts \
    && chown -R qts:qts /app /var/lib/qts /mlflow

USER qts

CMD ["python", "-m", "qts.orchestration.serve"]
