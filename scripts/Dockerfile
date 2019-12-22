FROM pytorch/pytorch


ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

ADD ./scripts/index_data.py index_data.py
ADD ./src src
ADD ./data/xsmall ./data

CMD ["python", "index_data.py", "-d", "./data"]