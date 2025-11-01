FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "sps_genai.sps_genai_assignment3.main_fastapi_gan:app", "--host", "0.0.0.0", "--port", "8000"]
