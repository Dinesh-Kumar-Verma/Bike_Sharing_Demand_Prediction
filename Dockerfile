# Use an official Python 3.10 image from Docker Hub
FROM python:3.13.5-slim

# Add this line to install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . .

# Install the dependencies
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
RUN pip install -r requirements_docker.txt

# Pull the DVC-tracked files
RUN dvc pull --force

#RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Streamlit default is 8501)
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]  