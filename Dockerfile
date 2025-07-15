FROM python:3.12-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends git
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
