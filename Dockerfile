# =====================================================
# Dockerfile — VetSkin AI
# Target: Hugging Face Spaces (Docker runtime)
# =====================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem yang diperlukan TensorFlow & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy dan install Python dependencies lebih dulu (memanfaatkan Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code project
COPY . .

# Expose port yang digunakan Hugging Face Spaces
EXPOSE 7860

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
