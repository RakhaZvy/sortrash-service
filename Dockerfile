# Use Python 3.9
FROM python:3.9-slim

# Install system dependencies for OpenCV (GL libraries)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a writable directory for temporary file uploads if needed
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run with Gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]