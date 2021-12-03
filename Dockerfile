FROM tensorflow:latest-gpu

RUN apt install git

RUN git clone git@github.com:bountrisv/dis-exercise-4.git
