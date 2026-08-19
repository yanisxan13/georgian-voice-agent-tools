FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Shell form so ${PORT} expands. Render injects PORT; 8000 is the local default.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
