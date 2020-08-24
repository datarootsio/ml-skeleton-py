FROM python:3.7
RUN apt-get update && apt-get install -y python-dev libffi-dev build-essential
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
ENV LOG_LEVEL INFO
ENTRYPOINT ["make", "test-package"]
