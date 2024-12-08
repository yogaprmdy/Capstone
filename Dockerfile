# Gunakan base image Python 3.9-slim
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory dalam container
WORKDIR /app

# Salin file requirements.txt ke WORKDIR
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dan folder ke WORKDIR
COPY . /app/

# Expose port untuk FastAPI
EXPOSE 8080

# Jalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
