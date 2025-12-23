# Use Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# "libgl1-mesa-glx" is replaced by "libgl1" in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code
COPY . .

# Create a writable directory for uploads
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose port 7860
EXPOSE 7860

# Start the application
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]