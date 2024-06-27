# Use the Python 3.12 image as the base image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Activate the virtual environment and install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Specify the command to run the application
CMD ["streamlit", "run", "app.py"]
