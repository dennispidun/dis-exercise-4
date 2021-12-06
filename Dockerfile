FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN mkdir -p /usr/bin/app

RUN cd /usr/bin/app

RUN git clone https://github.com/bountrisv/dis-exercise-4.git

ENTRYPOINT ["sleep", "infinity"]
