FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required to build many Python packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential gcc libpq-dev curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Create a non-root user and set ownership
RUN useradd -m appuser \
	&& chown -R appuser:appuser /app

USER appuser

# Default port (Flask default). Adjust if your app uses another port.
EXPOSE 8000

# Default command. If you use gunicorn or a different entrypoint, replace this.
CMD ["python", "app.py"]
