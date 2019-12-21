FROM pytorch/pytorch

RUN pip install flask
RUN pip install flask_cors

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

ADD . /app
WORKDIR /app
