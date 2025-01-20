FROM python:3.12-alpine3.18
RUN apk add --no-cache git nano
RUN git clone https://github.com/procrastinando/kids-math-homework /app
WORKDIR /app
RUN pip install --no-cache-dir gradio
CMD ["python", "/app/app.py"]