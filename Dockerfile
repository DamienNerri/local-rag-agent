FROM python:3.11-slim

# OCR des PDF : Tesseract (modèle français) + Poppler pour pdf2image
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Les documents sont montés en volume (voir docker-compose.yml), jamais copiés dans l'image
CMD ["python", "-u", "main.py"]
