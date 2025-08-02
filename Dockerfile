# 1. Select a suitable base image from Google Cloud
FROM gcr.io/tpu-pytorch/tpu-vm-base:tpu-nightly-v6e

# Set the working directory
WORKDIR /app

# 2. Install Python dependencies
# Copy requirements first to leverage Docker layer caching
COPY tensor_crunch/requirements.txt .

# Install git for the qwix dependency
RUN apt-get update && apt-get install -y git

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install safetensors

# 3. Configure Kaggle API credentials
# Use ARG to allow build-time variables and ENV to set them in the container
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY

# 4. Copy necessary application files
COPY tensor_crunch/gemma3_1b_medical_qa_fft_trainer_vllm_refactored.py .
COPY tensor_crunch/data.py .

# 5. Define the CMD to execute the Python script
ENTRYPOINT ["python", "gemma3_1b_medical_qa_fft_trainer_vllm_refactored.py"]
CMD ["--help"]