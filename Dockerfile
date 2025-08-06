# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim

# Add this line to install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . .

# Install the dependencies
RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Streamlit default is 8501)
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]    