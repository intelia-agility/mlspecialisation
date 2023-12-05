FROM us-central1-docker.pkg.dev/blackfridayintelia/quickstart-docker-repo/fastai_0

WORKDIR /app
COPY . /app

RUN echo "The AIP_STORAGE_URI is $AIP_STORAGE_URI"
ENTRYPOINT gunicorn -b :5080 app:app --timeout 60