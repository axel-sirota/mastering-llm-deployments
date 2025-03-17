# Dockerfile for Hugging Face Space deployment
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py app.py
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
