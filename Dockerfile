FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Сборочные зависимости dlib + runtime-библиотеки OpenCV + espeak для TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        libopenblas-dev liblapack-dev \
        libgl1 libglib2.0-0 \
        espeak-ng libespeak1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости копируем отдельно от кода ради кэширования слоя:
# пока requirenments.txt не меняется, долгая сборка dlib берётся из кэша.
COPY requirenments.txt .
# opencv-python заменяем на headless-сборку: GUI в контейнере не нужен
RUN grep -viE '^opencv-python([^-]|$)' requirenments.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt "opencv-python-headless>=4.5.0"

COPY *.py ./
COPY pytest.ini ./
COPY tests/ ./tests/

EXPOSE 5555 5556 8080

CMD ["python", "server.py", "--no-debug"]
