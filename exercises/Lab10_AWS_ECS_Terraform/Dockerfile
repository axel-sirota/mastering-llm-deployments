# Dockerfile for AWS ECS Deployment (Lab 10)
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py app.py
EXPOSE 7860
CMD ["python", "app.py"]
