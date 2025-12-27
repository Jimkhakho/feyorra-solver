FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install -y libgl1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
