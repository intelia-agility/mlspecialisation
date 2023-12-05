FROM python:3.9

ENV AIP_STORAGE_URI=default
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
