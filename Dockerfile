FROM python:3.10-slim-bullseye

WORKDIR /app

# 1. Install system build dependencies
# We keep them in a single layer to make cleanup effective later
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. OPTIMIZATION: Install PyTorch CPU-only first
# This prevents EasyOCR from downloading the massive NVIDIA CUDA version (saves ~5GB)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Copy and install other requirements
COPY dependencies.txt .
# We exclude easyocr from requirements.txt logic here to ensure it uses our pre-installed torch
RUN pip install --no-cache-dir -r dependencies.txt

# 4. Remove heavy build tools now that libraries are installed
# This deletes gcc and other compilers to save space
RUN apt-get purge -y build-essential && apt-get autoremove -y

# 5. Pre-download NLTK data (same as before)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('gutenberg'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# 6. Pre-download EasyOCR model
RUN python -c "import easyocr; easyocr.Reader(['vi'])"

COPY . .

# Ensure data directory exists
RUN mkdir -p data/imgs

# OLD COMMAND:
# CMD ["python", "server.py"]

# NEW PRODUCTION COMMAND:
# -w 1: Use 1 worker (Safest for heavy OCR tasks to prevent running out of RAM)
# --threads 2: Allow concurrent requests without crashing memory
# -b 0.0.0.0:5000: Bind to all interfaces on port 5000
# --timeout 120: Wait 120s before giving up (OCR can be slow)
# server:app : This means "Look in server.py for the object named app"
CMD ["gunicorn", "-w", "1", "--threads", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "server:app"]