FROM python:3.13.5-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install DVC (with S3 support if needed)
RUN pip install --no-cache-dir dvc[s3]

# Copy project files including DVC metadata
COPY . .

# Pull DVC data using AWS credentials from secret
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    dvc pull --force

# Expose port for app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]  