FROM python:3.10-slim
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
ENV PYTHONBUFFERED=1 PYTHONIOENCODING=utf-8ENV PYTHONIOENCODING=utf-8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8080

# CMD gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 wsgi:app
# Temporary
CMD python wsgi.py
