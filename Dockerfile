# Use an official Python 3.10 image from Docker Hub
FROM python:3.13.5-slim

# Add this line to install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . .

# Install the dependencies
RUN pip install -r requirements_docker.txt

# Set up AWS credentials for DVC
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
RUN mkdir -p ~/.aws && \
    echo "[default]" > ~/.aws/credentials && \
    echo "aws_access_key_id = $AWS_ACCESS_KEY_ID" >> ~/.aws/credentials && \
    echo "aws_secret_access_key = $AWS_SECRET_ACCESS_KEY" >> ~/.aws/credentials && \
    echo "[default]" > ~/.aws/config && \
    echo "region = $AWS_DEFAULT_REGION" >> ~/.aws/config

# Pull the DVC-tracked files
RUN dvc pull --force

#RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Streamlit default is 8501)
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]  