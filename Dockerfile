FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    libbz2-dev

RUN mkdir -p /usr/bin/app

WORKDIR /usr/bin/app

COPY . dis-exercise-4

WORKDIR /usr/bin/app/dis-exercise-4

# Soon to become deprecated and be replaced by:
# curl -sSL https://install.python-poetry.org | python3 -
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

RUN /root/.poetry/bin/poetry export -f requirements.txt --without-hashes > requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["sleep", "infinity"]
