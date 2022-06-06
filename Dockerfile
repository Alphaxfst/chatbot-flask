FROM python:3.10-slim
ENV PYTHONBUFFERED=1

RUN apt-get update 


RUN pip install pipenv
COPY Pipfile /app/

WORKDIR /app

RUN pipenv lock --keep-outdated --requirements > requirements.txt && \
	pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
COPY . .

ENTRYPOINT ["gunicorn", "--reload", "--bind", ":5000", "wsgi:app"]
