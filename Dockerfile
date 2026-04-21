FROM python:3.13-slim

# Ustawienia srodowiska
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Katalog roboczy
WORKDIR /app

# Kopiujemy najpierw requirements.txt, aby wykorzystac cache warstw Dockera
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Kopiujemy pozostale pliki aplikacji
COPY model.py app.py ./
COPY tests/ ./tests/

# Port, na ktorym dziala FastAPI
EXPOSE 8000

# Uruchomienie serwera uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
