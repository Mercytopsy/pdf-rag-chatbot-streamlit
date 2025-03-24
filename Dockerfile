# Use an official Python image
FROM python:3.12

# Set working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    postgresql postgresql-client \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501


ENTRYPOINT ["bash", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]



