FROM python:3.10-slim
ENV PYTHONBUFFERED=1

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", ":$PORT", "--timeout", "0" "wsgi:app"]
