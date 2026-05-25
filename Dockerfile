FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models && chmod 777 /app/models

EXPOSE 5000

CMD gunicorn -b 0.0.0.0:${PORT:-5000} app:app --timeout 120 --workers 1
