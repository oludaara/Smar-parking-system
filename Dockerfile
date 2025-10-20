# Use a Python image
FROM python:3.10

# Install system dependencies for OpenCV, Tesseract, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run your app using Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "server:app"]
