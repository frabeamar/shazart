# Use a slim version of Python for a smaller image size
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for some Python packages (like pymilvus/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency file first to leverage Docker cache
# Since you provided a PEP 621 style [project] block, 
# we'll assume you're using a pyproject.toml
COPY pyproject.toml .

# Install dependencies
# We use pip to install the requirements directly from the pyproject.toml
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the container is running properly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
