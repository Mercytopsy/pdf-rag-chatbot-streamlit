# Use an official Python image
FROM python:3.12

# Set working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install system dependencies
RUN apt update && apt install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Redis
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(grep VERSION_CODENAME /etc/os-release | cut -d= -f2) main" | \
    tee /etc/apt/sources.list.d/redis.list && \
    apt-get update && \
    apt-get install -y redis-stack-server

# Start PostgreSQL and Redis
RUN service postgresql start && \
    redis-stack-server --daemonize yes

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# # Set the default command to run the app
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Set the default command to run the app
# CMD ["streamlit", "run", "app.py", "--server.port=${PORT:-8501}", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

ENTRYPOINT ["bash", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
