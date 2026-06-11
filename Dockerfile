FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

ENTRYPOINT ["python", "src/inference.py"]
CMD ["--input", "data/sample_batch.csv", "--output", "artifacts/predictions.csv"]
