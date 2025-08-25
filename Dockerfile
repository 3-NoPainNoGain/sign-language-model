FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU용 PyTorch/torchvision (버전쌍 중요)
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
