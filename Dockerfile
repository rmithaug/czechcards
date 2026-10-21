# Czech Cards in a container.
#   Build:  docker build -t czech-cards .
#   Run:    docker run --rm -p 8000:8000 czech-cards
#   Play:   http://localhost:8000

FROM python:3.13-slim

# Don't write .pyc files, and print logs straight away.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=0

WORKDIR /app

# Install the libraries first, so Docker can reuse this step when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a normal user instead of root.
RUN useradd --create-home player
USER player

EXPOSE 8000
CMD ["python", "app.py"]