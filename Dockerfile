FROM python:3.11-slim

WORKDIR /app

# Install system dependencies sumolib needs
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Create output directories
RUN mkdir -p outputs/graph_reconstruction outputs/networks

EXPOSE 7860

CMD ["sh", "-c", "python download_data.py && gunicorn route_app:app --workers 1 --timeout 300 --bind 0.0.0.0:7860"]
