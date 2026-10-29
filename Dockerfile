FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the model into the image so containers start without a network download.
ARG MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL}', device='cpu', compute_type='int8')"
COPY transcriber ./transcriber
ENV TRANSCRIBER_MODEL=${MODEL} TRANSCRIBER_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "transcriber", "serve", "--host", "0.0.0.0", "--port", "8000"]
