FROM python:3.10-slim
ENV PYTHONBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
COPY . .

ENTRYPOINT ["gunicorn", "--bind", ":$PORT", "--timeout", "0" "wsgi:app"]
