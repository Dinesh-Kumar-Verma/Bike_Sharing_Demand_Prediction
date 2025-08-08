# Stage 1: Build and pull DVC artifacts
FROM python:3.13.5-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements_docker.txt

# Pull DVC artifacts using a secret mount
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials dvc pull --force

# Stage 2: Final application image
FROM python:3.13.5-slim

WORKDIR /app

# Copy application code and downloaded artifacts from the builder stage
COPY --from=builder /app/ . 

# Expose port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]  