FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    joblib==1.3.2 \
    flask==3.0.0

# Copy model files
COPY Heart_Disease_RandomForest_model.pkl /app/
COPY scaler.pkl /app/
COPY inference_serving.py /app/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run inference server
CMD ["python", "inference_serving.py"]
