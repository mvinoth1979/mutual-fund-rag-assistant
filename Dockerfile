# Use Python 3.12 slim as base
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Copy backend requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Build the frontend
WORKDIR /app/phase_7_frontend
RUN npm install
RUN npm run build

# Move back to root
WORKDIR /app

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY . .

# Copy built frontend assets to a directory FastAPI can serve
COPY --from=builder /app/phase_7_frontend/dist /app/static

# Expose the port Railway uses (default 8080 or $PORT)
EXPOSE 8080

# Run the application using the port provided by Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
