FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY scripts/generate_example_data.py /app/scripts/generate_example_data.py
COPY docker/backend_entrypoint.sh /app/backend_entrypoint.sh

ENV PYTHONPATH=/app
ENV CSO_DATA_ROOT=/app/data

RUN chmod +x /app/backend_entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/backend_entrypoint.sh"]
