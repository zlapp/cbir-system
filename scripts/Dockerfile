FROM pytorch/pytorch

RUN pip install --user annoy

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD ./scripts/index_data.py index_data.py
ADD ./src src
ADD ./data/small ./data
ADD ./configs/v12annoy.cfg config.cfg

CMD ["python", "index_data.py", "-d", "./data", "-c", "config.cfg"]