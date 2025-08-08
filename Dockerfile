FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with S3 support
RUN pip install --no-cache-dir dvc[s3]

# Copy project files
COPY . .

# Pull DVC data (AWS creds via secret mount in GitHub Actions)
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    dvc pull --force

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
